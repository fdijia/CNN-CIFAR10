import json

import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt

activation_options = {
    'relu': nn.ReLU,                # {'activation': {'type': 'relu', 'inplace': False}}
    'leaky_relu': nn.LeakyReLU,     # {'activation': {'type': 'leaky_relu', 'negative_slope': 0.1, 'inplace': False}}
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'selu': nn.SELU,
    'gelu': nn.GELU,                # {'activation': {'type': 'gelu', 'approximate': 'tanh'}}
    'swish': lambda: nn.SiLU(),     # {'activation': {'type': 'swish'}}
    'mish': lambda: nn.Mish(),
    'hardswish': nn.Hardswish
}

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class ConfigurableCNN(nn.Module):
    def __init__(self, config):
        super(ConfigurableCNN, self).__init__()
        self.layers = self._build_layers(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_layers(self, config):
        """根据配置动态构建网络层"""
        layers = OrderedDict()
        for i, layer_config in enumerate(config):
            layer_name, layer_params = next(iter(copy.deepcopy(layer_config).items()))

            if layer_name == 'conv':
                layer = nn.Conv2d(**layer_params)
            elif layer_name == 'bn':
                layer = nn.BatchNorm2d(**layer_params)
            elif layer_name == 'pool':
                layer_type = layer_params.pop('type', 'max')
                layer = nn.MaxPool2d(**layer_params) if layer_type == 'max' else nn.AvgPool2d(**layer_params)
            elif layer_name == 'flatten':
                layer = nn.Flatten()
            elif layer_name == 'fc':
                layer = nn.Linear(**layer_params)
            elif layer_name == 'dropout':
                layer_type = layer_params.pop('type', '2d')
                layer = nn.Dropout(**layer_params) if layer_type == '1d' else nn.Dropout2d(**layer_params)
            elif layer_name == 'activation':
                act_type = layer_params.pop('type', 'relu')  # 取出 'type'，剩下的参数用于初始化
                act_class = activation_options[act_type]
                if act_type == 'leaky_relu' and 'negative_slope' not in layer_params:
                    layer_params['negative_slope'] = 0.01
                layers[f"{act_type}_{i}"] = act_class(**layer_params)
                continue
            elif layer_name == 'residual':
                layer = ResidualBlock(**layer_params)
            else:
                raise ValueError(f"Unknown layer type: {layer_name}")

            layers[f"{layer_name}_{i}"] = layer

        return nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)

    def visualize_conv_layers(self, layer_indices=None, max_columns=16, save_path=None):
        """
        :param layer_indices: 要可视化的卷积层索引列表，None表示所有卷积层
        :param max_columns: 每次最多显示多少个列
        """
        conv_layers = []
        # 收集所有卷积层
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                conv_layers.append((i, layer))

        if not conv_layers:
            print("No convolutional layers found in the model.")
            return

        # 如果指定了层索引，只显示这些层
        if layer_indices is not None:
            conv_layers = [cl for idx, cl in enumerate(conv_layers) if idx in layer_indices]

        for layer_idx, layer in conv_layers:
            weights = layer.weight.detach().cpu().numpy()
            # 权重形状: (out_channels, in_channels, kernel_h, kernel_w)

            # 归一化权重到0-1
            min_val, max_val = weights.min(), weights.max()
            weights = (weights - min_val) / (max_val - min_val + 1e-8)

            out_channels, in_channels = weights.shape[:2]
            num_columns = min(max_columns, out_channels)
            num_rows = (out_channels - 1) // num_columns + 1

            fig, axes = plt.subplots(num_rows * in_channels, num_columns,
                                     figsize=(num_columns * 1.5, num_rows * in_channels * 1.5))
            for in_ch in range(in_channels):
                for i in range(num_rows):
                    for j in range(num_columns):
                        filter_idx = i * num_columns + j
                        row_idx = in_ch + i * in_channels
                        ax = axes[row_idx, j]

                        ax.imshow(weights[filter_idx, in_ch], cmap='gray')
                        ax.set_xticks([])
                        ax.set_yticks([])

                        if j == 0:
                            ax.set_ylabel(f'in {in_ch + 1}', rotation=0, ha='right', va='center')
                        if in_ch == 0:
                            ax.set_title(f'out {filter_idx + 1}')

            plt.suptitle(f'Conv Layer {layer_idx}')
            plt.tight_layout()
            if save_path is not None:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

    def get_loss_landscape(self, dataloader, criterion, directions=None, steps=25, range_=(-1, 1)):
        """
        Calculate loss landscape using a DataLoader
        :param dataloader: DataLoader providing batches of (X, y)
        :param criterion: Loss function
        :param directions: Direction vectors (two directions), None for random
        :param steps: Number of steps in each direction
        :param range_: Exploration range
        :return: alphas, betas, losses (for plotting)
        """
        original_weights = [p.detach().clone() for p in self.parameters()]
        model = self.to(self.device)
        # Generate random directions if not provided
        if directions is None:
            directions = []
            for _ in range(2):
                direction = [torch.randn_like(p) for p in original_weights]
                # Normalize directions
                norm = torch.sqrt(sum(torch.sum(d ** 2) for d in direction))
                direction = [d / norm for d in direction]
                directions.append(direction)

        alphas = torch.linspace(range_[0], range_[1], steps)
        betas = torch.linspace(range_[0], range_[1], steps)
        losses = torch.zeros(steps, steps)

        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    print('start', i, j)
                    # Move parameters along directions
                    for p, orig, d1, d2 in zip(model.parameters(), original_weights, directions[0], directions[1]):
                        p.copy_(orig + alpha * d1 + beta * d2)

                    # Calculate loss over entire dataset
                    total_loss = 0.0
                    num_batches = 0

                    for X, y in dataloader:
                        X, y = X.to(self.device), y.to(self.device)

                        outputs = model(X)
                        loss = criterion(outputs, y)
                        total_loss += loss.item()
                        num_batches += 1

                    losses[i, j] = total_loss / num_batches

        return alphas.numpy(), betas.numpy(), losses.numpy()

    def plot_loss_landscape(self, dataloader, criterion, directions=None, steps=25, range_=(-1, 1), save_path=None):
        alphas, betas, losses = self.get_loss_landscape(dataloader, criterion, directions, steps, range_)
        save_history = {
            'alphas': alphas.tolist(),
            'betas': betas.tolist(),
            'losses': losses.tolist()
        }
        with open(save_path + '/landscape.json', 'w') as f:
            json.dump(save_history, f)

        plt.figure(figsize=(10, 8))
        plt.contourf(alphas, betas, losses, levels=20, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Direction 1 (alpha)')
        plt.ylabel('Direction 2 (beta)')
        plt.title('2D Loss Landscape')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

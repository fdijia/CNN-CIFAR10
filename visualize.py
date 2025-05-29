import seaborn as sns
import os
from typing import Dict, List, Optional
from model import ConfigurableCNN
import importlib.util
from pathlib import Path
import torch
import json
from config import ReadConfig
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def plot_training_curves(self, history: Dict, save: bool = True):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        epochs = range(1, len(history['val_loss']) + 1)
        steps = range(1, len(history['train_loss']) + 1)
        plot_dict = [[
            {'x': steps, 'y': history['train_loss'], 'title': 'Train Loss', 'xlabel': 'Step', 'ylabel': 'Loss'},
            {'x': epochs, 'y': history['val_loss'], 'title': 'Validation Loss', 'xlabel': 'Epoch', 'ylabel': 'Loss'},
            {'x': epochs, 'y': history['learning_rate'], 'title': 'Learning Rate', 'xlabel': 'Epoch', 'ylabel': 'Learning Rate'},
        ],[
            {'x': steps, 'y': history['train_acc'], 'title': 'Train Accuracy', 'xlabel': 'Step', 'ylabel': 'Accuracy (%)'},
            {'x': epochs, 'y': history['val_acc'], 'title': 'Validation Accuracy', 'xlabel': 'Epoch', 'ylabel': 'Accuracy (%)'},
            {'x': epochs, 'y': history['epoch_time'], 'title': 'Epoch Time', 'xlabel': 'Epoch', 'ylabel': 'Time (s)'},
        ]]

        for i in range(2):
            for j in range(3):
                axes[i, j].plot(plot_dict[i][j]['x'], plot_dict[i][j]['y'])
                axes[i, j].set_title(plot_dict[i][j]['title'])
                axes[i, j].set_xlabel(plot_dict[i][j]['xlabel'])
                axes[i, j].set_ylabel(plot_dict[i][j]['ylabel'])
                axes[i, j].grid(True, alpha=0.3)

        axes[0, 2].set_yscale('log')

        plt.tight_layout()
        if save:
            plt.savefig(self.results_dir + '/curves.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def compare_experiments(self, experiment_results: Dict[str, Dict], metrics, save_name=None):
        sns.set_palette("colorblind")
        num_metrics = len(metrics)
        fig, axs = plt.subplots(2, num_metrics, figsize=(5*num_metrics, 15))
        if num_metrics == 1:
            axs = axs.reshape(2, 1)
        for i, metric in enumerate(metrics):
            # 第一行：训练曲线对比
            ax_curve = axs[0, i] if num_metrics > 1 else axs[0]
            for exp_name, results in experiment_results.items():
                history = results['history']
                epochs = range(1, len(history[metric]) + 1)
                ax_curve.plot(epochs, history[metric], label=exp_name, linewidth=2)

            ax_curve.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax_curve.set_xlabel('Epoch')
            ax_curve.set_ylabel(metric.replace("_", " ").title())
            ax_curve.legend()
            ax_curve.grid(True, alpha=0.3)

            # 第二行：最终性能柱状图
            ax_bar = axs[1, i] if num_metrics > 1 else axs[1]
            exp_names = list(experiment_results.keys())
            final_scores = [experiment_results[name]['history'][metric][-1] for name in exp_names]
            ax_title = f'Final {metric.replace("_", " ").title()}'

            bars = ax_bar.bar(exp_names, final_scores)
            ax_bar.set_title(ax_title)
            ax_bar.set_ylabel(metric.replace("_", " ").title())
            ax_bar.tick_params(axis='x', rotation=45)

            # 在柱子上添加数值标签
            for bar, score in zip(bars, final_scores):
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                            f'{score:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        if save_name:
            plt.savefig(self.results_dir + '/' + save_name, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class TestingVisualizer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.config = self.get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = self.get_checkpoint()
        self.model = ConfigurableCNN(config=self.config['model'])
        self.model.load_state_dict(self.checkpoint)
        self.results = self.get_results()

    def get_config(self):
        path = self.results_dir + '/config.py'
        abs_path = Path(path).resolve()

        spec = importlib.util.spec_from_file_location("temp_module", abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'config'):
            return getattr(module, 'config')
        else:
            raise AttributeError(f"config未找到")

    def get_checkpoint(self):
        path = self.results_dir + '/final.pth'
        return torch.load(path, weights_only=False, map_location=self.device)

    def get_results(self):
        path = self.results_dir + '/results.json'
        return json.load(open(path))

    def visualize_conv(self):
        self.model.visualize_conv_layers(layer_indices=[0], save_path=self.results_dir + '/conv.png')


    def visualize_landscape(self):
        criterion = ReadConfig.get_criterion(self.config)
        loader = ReadConfig.get_loaders(self.config)
        self.model.plot_loss_landscape(loader[0], criterion, save_path=self.results_dir)


if __name__ == '__main__':
    visualizer = TestingVisualizer(results_dir='results/cnn25')
    visualizer.visualize_conv()
    visualizer.visualize_landscape()
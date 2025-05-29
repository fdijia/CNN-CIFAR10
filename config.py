# opti: sgd, adam, adamw (need init_lr, weight_decay)
# scheduler: step(scheduler_step_size, scheduler_gamma), cosine, multistep(milestones, scheduler_gamma)
# criterion: CE, LabelSmoothing(smoothing), Focal(focal_gamma)
# use_mixup: True(mixup_alpha), False
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from loss import FocalLoss, LabelSmoothingLoss
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets

def get_best_config():
    config = deepcopy(WeightDecay.get_best_config())
    config.update({'milestones': [12, 18, 23], 'num_epochs': 25, 'model_name': 'cnn25'})
    return config

def get_resnet18():
    config = get_best_config()
    config.update(Filters.res18_config)
    config.update({'model_name': 'res18'})
    return config

class Filters:
    basic_config = {
        'opti': 'sgd',
        'scheduler': 'multistep',
        'milestones': [8, 12, 14],
        'scheduler_gamma': 0.2,
        'init_lr': 0.01,
        'weight_decay': 1e-4,
        'criterion': 'CE',
        'use_mixup': False,
        'num_epochs': 15,
        'save_model': True,
        'log_interval': 50,
    }

    three_layer_cnn_config = {
        'model': [
            {'conv': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 64}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}}, # 16x16x64

            {'conv': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 128}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}}, # 8x8x128

            {'conv': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 256}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'avg', 'kernel_size': 8}}, # 1x1x256
            {'flatten': {}},
            {'fc': {'in_features': 256, 'out_features': 10}}
        ],
        'model_name': 'three layer cnn'
    }

    four_layer_cnn_config = {
        'model': [
            {'conv': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 64}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}},

            {'conv': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 128}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}},

            {'conv': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 256}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}},

            {'conv': {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 512}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'avg', 'kernel_size': 4}},
            {'flatten': {}},
            {'fc': {'in_features': 512, 'out_features': 10}}
        ],
        'model_name': 'four layer cnn'
    }

    five_layer_cnn_config = {
        'model': [
            {'conv': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 64}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}},

            {'conv': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 128}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}},

            {'conv': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 256}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 2, 'stride': 2}},

            {'conv': {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 512}},
            {'activation': {'type': 'relu'}},

            {'conv': {'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1}},
            {'bn': {'num_features': 512}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'avg', 'kernel_size': 4}},

            {'flatten': {}},
            {'fc': {'in_features': 512, 'out_features': 10}}
        ],
        'model_name': 'five layer cnn'
    }

    residual1_cnn_config = {
        'model': [
            {'conv': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
            {'bn': {'num_features': 64}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 64, 'out_channels': 128, 'stride': 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 128, 'out_channels': 256, 'stride': 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'pool': {'type': 'avg', 'kernel_size': 4}},
            {'flatten': {}},
            {'fc': {'in_features': 256, 'out_features': 10}}
        ],
        'model_name': 'residual1_cnn',
    }

    residual2_cnn_config = {
        'model': [
            {'conv': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
            {'bn': {'num_features': 64}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 64, 'out_channels': 128, 'stride': 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 128, 'out_channels': 256, 'stride': 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 256, 'out_channels': 256, 'stride': 1}},
            {'activation': {'type': 'relu'}},

            {'pool': {'type': 'avg', 'kernel_size': 4}},

            {'flatten': {}},
            {'fc': {'in_features': 256, 'out_features': 10}},],
        'model_name': 'residual2_cnn',
    }

    residual3_cnn_config = {
        'model': [
            {'conv': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
            {'bn': {'num_features': 64}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 64, 'out_channels': 128, 'stride': 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 128, 'out_channels': 256, 'stride': 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},

            {'residual': {'in_channels': 256, 'out_channels': 512, 'stride': 1}},
            {'activation': {'type': 'relu'}},

            {'pool': {'type': 'avg', 'kernel_size': 4}},

            {'flatten': {}},
            {'fc': {'in_features': 512, 'out_features': 10}},],
        'model_name': 'residual3_cnn'
    }

    res18_config = {
        'model': [
            {"conv": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"bn": {"num_features": 64}},
            {"activation": {"type": "relu"}},

            {"residual": {"in_channels": 64, "out_channels": 64, "stride": 1}},
            {"residual": {"in_channels": 64, "out_channels": 64, "stride": 1}},

            {"residual": {"in_channels": 64, "out_channels": 128, "stride": 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},
            {"residual": {"in_channels": 128, "out_channels": 128, "stride": 1}},

            {"residual": {"in_channels": 128, "out_channels": 256, "stride": 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},
            {"residual": {"in_channels": 256, "out_channels": 256, "stride": 1}},

            {"residual": {"in_channels": 256, "out_channels": 512, "stride": 1}},
            {'activation': {'type': 'relu'}},
            {'pool': {'type': 'max', 'kernel_size': 3, 'stride': 2, 'padding': 1}},
            {"residual": {"in_channels": 512, "out_channels": 512, "stride": 1}},

            {"pool": {"type": "avg", "kernel_size": 4}},
            {"flatten": {}},
            {"fc": {"in_features": 512, "out_features": 10}}],
        'model_name': 'res18_cnn',
    }

    @staticmethod
    def get_config():
        basic_config = Filters.basic_config
        configs = [deepcopy(Filters.three_layer_cnn_config),
                   deepcopy(Filters.four_layer_cnn_config),
                   deepcopy(Filters.five_layer_cnn_config),
                   deepcopy(Filters.residual1_cnn_config),
                   deepcopy(Filters.residual2_cnn_config),
                   deepcopy(Filters.residual3_cnn_config),
                   deepcopy(Filters.res18_config)]
        for config in configs:
            config.update(basic_config)
        return configs

    @staticmethod
    def get_best_config():
        config = Filters.residual3_cnn_config
        config.update(Filters.basic_config)
        return deepcopy(config)

class Criterion:
    basic_config = deepcopy(Filters.basic_config)
    basic_config.update(Filters.residual3_cnn_config)
    criterions = [
        {'model_name': 'CE'},
        {'criterion': 'CE', 'use_mixup': True, 'mixuo_alpha': 0.1, 'model_name': 'CE_mixup0.1'},
        {'criterion': 'CE', 'use_mixup': True, 'mixuo_alpha': 0.2, 'model_name': 'CE_mixup0.2'},
        {'criterion': 'CE', 'use_mixup': True, 'mixuo_alpha': 0.3, 'model_name': 'CE_mixup0.3'},
        {'criterion': 'CE', 'use_mixup': True, 'mixuo_alpha': 0.4, 'model_name': 'CE_mixup0.4'},
        {'criterion': 'LabelSmoothing', 'smoothing': 0.02, 'model_name': 'LabelSmoothing'},
        {'criterion': 'Focal', 'focal_gamma': 2.0, 'model_name': 'Focal'},
    ]

    @staticmethod
    def get_config():
        configs = []
        for criterion in Criterion.criterions:
            config = deepcopy(Criterion.basic_config)
            config.update(criterion)
            configs.append(config)
        return configs

    @staticmethod
    def get_best_config():
        config = Filters.get_best_config()
        config.update({'criterion': 'CE', 'use_mixup': True, 'mixuo_alpha': 0.1, 'model_name': 'CE_mixup0.1'})
        return deepcopy(config)

class Activations:
    basic_config = Criterion.get_best_config()
    model_config = deepcopy(Filters.residual3_cnn_config['model']) # a list
    activations = (
        {'relu': {'activation': {'type': 'relu'}}},
        {'leaky_relu_0.01': {'activation': {'type': 'leaky_relu', 'negative_slope': 0.01}}},
        {'leaky_relu_0.02': {'activation': {'type': 'leaky_relu', 'negative_slope': 0.02}}},
        {'swish': {'activation': {'type': 'swish'}}},
        {'gelu': {'activation': {'type': 'gelu'}}},
    )

    @staticmethod
    def get_config():
        configs = []
        for activation in Activations.activations:
            model_name, layer = next(iter(activation.items()))
            network_config = deepcopy(Activations.model_config)
            for i in range(len(network_config)):
                layer_name, layer_params = next(iter(network_config[i].items()))
                if layer_name == 'activation':
                    network_config[i] = layer
            config = deepcopy(Activations.basic_config)
            config['model'] = network_config
            config['model_name'] = model_name
            configs.append(config)
        return configs

    @staticmethod
    def get_best_config():
        config = Criterion.get_best_config()
        network_config = config['model']
        for i in range(len(network_config)):
            layer_name, layer_params = next(iter(network_config[i].items()))
            if layer_name == 'activation':
                network_config[i] = {'activation': {'type': 'gelu'}}
        config['model_name'] = 'gelu'
        return deepcopy(config)

class Optim:
    basic_config = Activations.get_best_config()
    optis = [
        {'opti': 'sgd', 'init_lr': 0.01, 'model_name': 'SGD0.01'},
        {'opti': 'sgd', 'init_lr': 0.02, 'model_name': 'SGD0.02'},
        {'opti': 'sgd', 'init_lr': 0.005, 'model_name': 'SGD0.005'},
        {'opti': 'adam', 'init_lr': 0.002, 'model_name': 'adam0.002'},
        {'opti': 'adam', 'init_lr': 0.005, 'model_name': 'adam0.005'},
        {'opti': 'adam', 'init_lr': 0.001, 'model_name': 'adam0.001'},
        {'opti': 'adamw', 'init_lr': 0.002, 'model_name': 'adamw0.002'},
        {'opti': 'adamw', 'init_lr': 0.005, 'model_name': 'adamw0.005'},
        {'opti': 'adamw', 'init_lr': 0.001, 'model_name': 'adamw0.001'},
    ]

    @staticmethod
    def get_config():
        configs = []
        for opti in Optim.optis:
            config = deepcopy(Optim.basic_config)
            config.update(opti)
            configs.append(config)
        return configs

    @staticmethod
    def get_best_config():
        basic_config = Activations.get_best_config()
        basic_config.update({'opti': 'adamw', 'init_lr': 0.002, 'model_name': 'adamw0.002'})
        return basic_config

class WeightDecay:
    basic_config = Optim.get_best_config()
    weight_decays = [
        {'weight_decay': 1e-4, 'model_name': 'decay1e4'},
        {'weight_decay': 5e-4, 'model_name': 'decay5e4'},
        {'weight_decay': 1e-3, 'model_name': 'decay1e3'},
    ]

    @staticmethod
    def get_config():
        configs = []
        for weight_decay in WeightDecay.weight_decays:
            config = deepcopy(WeightDecay.basic_config)
            config.update(weight_decay)
            configs.append(config)
        return configs

    @staticmethod
    def get_best_config():
        config = Optim.get_best_config()
        config.update({'model_name': 'best cnn', 'weight_decay': 5e-4})
        return config

class ReadConfig:
    @staticmethod
    def get_criterion(config):
        if config['criterion'] == 'CE':
            return nn.CrossEntropyLoss()
        elif config['criterion'] == 'LabelSmoothing':
            return LabelSmoothingLoss(smoothing=config['smoothing'])
        elif config['criterion'] == 'Focal':
            return FocalLoss(gamma=config['focal_gamma'])
        else:
            raise ValueError(f"Unknown criterion: {config['criterion']}")

    @staticmethod
    def get_scheduler(config, optimizer):
        """Create learning rate scheduler based on configuration"""
        if config['scheduler'] == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'],
                                             gamma=config['scheduler_gamma'])
        elif config['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
        elif config['scheduler'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
        elif config['scheduler'] == 'multistep':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'],
                                                  gamma=config['scheduler_gamma'])
        else:
            raise ValueError(f"Unknown scheduler: {config['scheduler']}")
        
    @staticmethod
    def get_optimizer(config, model) -> optim.Optimizer:
        if config['opti'] == 'adam':
            return optim.Adam(model.parameters(), lr=config['init_lr'],
                            weight_decay=config['weight_decay'])
        elif config['opti'] == 'sgd':
            return optim.SGD(model.parameters(), lr=config['init_lr'],
                           momentum=0.9, weight_decay=config['weight_decay'])
        elif config['opti'] == 'adamw':
            return optim.AdamW(model.parameters(), lr=config['init_lr'],
                             weight_decay=config['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {config['opti']}")

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def get_cifar_dataset(root='data/', train=True):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if train:
            # Training transforms with augmentation
            data_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        return datasets.CIFAR10(root=root, train=train, transform=data_transforms, download=True)

    @staticmethod
    def get_loaders(config):
        datasets = ReadConfig.get_cifar_dataset()
        test_dataset = ReadConfig.get_cifar_dataset(train=False)
        batch_size = config.get('batch_size', 50)
        num_workers = config.get('num_workers', 4)
        train_ratio = config.get('train_ratio', 0.8)
        train_size = int(train_ratio * len(datasets))
        val_size = len(datasets) - train_size
        train_dataset, val_dataset = random_split(datasets, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True, persistent_workers=True)

        return [train_loader, val_loader, test_loader]


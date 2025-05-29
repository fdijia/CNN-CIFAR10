from main import run_single_experiment, run_comparison
from config import ReadConfig


def filter_find(loaders=None):
    from config import Filters
    configs = Filters.get_config()
    run_comparison(configs, save_name='filters.png', loaders=loaders, metrics=('train_acc', 'val_acc', 'test_acc', 'epoch_time'))

def criterion_find(loaders=None):
    from config import Criterion
    configs = Criterion.get_config()
    run_comparison(configs, save_name='criterion.png', loaders=loaders)


def activation_final(loaders=None):
    from config import Activations
    configs = Activations.get_config()
    run_comparison(configs, save_name='activation.png', loaders=loaders)

def optim_find(loaders=None):
    from config import Optim
    configs = Optim.get_config()
    run_comparison(configs, save_name='optim.png', loaders=loaders)

def weight_decay_find(loaders=None):
    from config import WeightDecay
    configs = WeightDecay.get_config()
    run_comparison(configs, save_name='weight_decay.png', loaders=loaders)


if __name__ == '__main__':
    config = {'batch_size': 100, 'num_workers': 4, 'train_ratio': 0.8}
    loaders = ReadConfig.get_loaders(config)
    import config
    configs = [config.get_best_config(), config.get_resnet18()]
    for c in configs:
        run_single_experiment(c, loaders, curve=True)
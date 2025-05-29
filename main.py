import torch
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple
from visualize import TrainingVisualizer
from model import ConfigurableCNN
from config import ReadConfig


class CIFAR10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = 'results/' + self.config['model_name']
        self.visualizer = TrainingVisualizer(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Experiment: {config['model_name']}")

    def mixup_data(self, x, y):
        alpha = self.config.get('mixup_alpha', 0.1)
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self, model, trainloader, optimizer, criterion, epoch):
        model.train()
        running_loss = []
        correct = []
        start_time = time.time()
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Apply mixup if enabled
            if self.config['use_mixup'] and np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = ReadConfig.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct.append(100 * (predicted == targets).sum().item() / targets.size(0))
            
            # Log progress
            if (i + 1) % self.config['log_interval'] == 0:
                print(f'Epoch [{epoch+1}/{self.config['num_epochs']}], '
                      f'Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')
        
        epoch_time = time.time() - start_time
        
        return running_loss, correct, epoch_time
    
    def test_epoch(self, model, testloader, criterion):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        test_acc = 100 * correct / total
        avg_loss = test_loss / len(testloader)
        
        return avg_loss, test_acc
    
    def train(self, loaders=None) -> Dict:
        """Main training loop"""
        if loaders:
            train_loader, val_loader, test_loader = loaders
        else:
            train_loader, val_loader, test_loader = ReadConfig.get_loaders(self.config)

        model = ConfigurableCNN(self.config['model'])
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {self.config['model_name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create optimizer and scheduler
        optimizer = ReadConfig.get_optimizer(self.config, model)
        scheduler = ReadConfig.get_scheduler(self.config, optimizer)
        criterion = ReadConfig.get_criterion(self.config)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        best_val_acc = 0.0
        
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc, epoch_time = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = self.test_epoch(model, val_loader, criterion)
            test_loss, test_acc = self.test_epoch(model, test_loader, criterion)

            scheduler.step()

            # Record history
            history['train_loss'].extend(train_loss)
            history['train_acc'].extend(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_time)
            
            print(f'Epoch [{epoch+1}/{self.config['num_epochs']}]: '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}% '
                  f'Time: {epoch_time:.2f}s')
        
        # Save final model
        if self.config['save_model']:
            torch.save(model.state_dict(), 
                     f'{self.results_dir}/final.pth')
        
        # Prepare results
        results = {
            'history': history,
            'val_acc': history['val_acc'][-1],
            'test_acc': history['test_acc'][-1],
            'model_parameters': total_params,
        }
        
        # Save results
        with open(f'{self.results_dir}/results.json', 'w') as f:
            json.dump(results, f, indent=2)

        from pprint import PrettyPrinter
        pp = PrettyPrinter(indent=4)
        formatted_config = pp.pformat(self.config)
        with open(f'{self.results_dir}/config.py', 'w') as f:
            f.write(f"config = {formatted_config}")
        
        print(f"\nTraining completed!")
        print(f"test accuracy: {results['test_acc']:.2f}%")
        
        return results


def run_single_experiment(config: dict, loaders: list = None, curve=False):
    trainer = CIFAR10Trainer(config)
    result_path = trainer.results_dir + '/results.json'
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            results = json.load(f)
    else:
        results = trainer.train(loaders=loaders)

    # Visualize results
    if curve:
        trainer.visualizer.plot_training_curves(results['history'], save=True)
    return results


def run_comparison(configs, save_name='filters.png', metrics=('train_acc', 'val_acc', 'test_acc'), loaders=None, curve=False):
    all_results = {}
    for config in configs:
        print(f"\n{'='*50}\nRunning {config['model_name']} experiment\n{'='*50}")
        results = run_single_experiment(config, loaders, curve)
        all_results[config['model_name']] = results
    visualizer = TrainingVisualizer(results_dir='visualization/comparison')
    visualizer.compare_experiments(all_results, metrics=metrics, save_name=save_name)
    return all_results

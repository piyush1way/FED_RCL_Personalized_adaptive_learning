import copy
import logging
import time
import gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils.logging_utils import AverageMeter
from utils.metrics import evaluate
from trainers.build import TRAINER_REGISTRY

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class BaseTrainer:
    def __init__(self, model=None, client_type=None, server=None, evaler_type=None, 
                datasets=None, device=None, args=None, config=None):
        self.model = model
        self.client_type = client_type
        self.server = server
        self.evaler_type = evaler_type
        self.datasets = datasets
        self.device = device
        self.args = args
        self.config = config
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        if hasattr(args, 'client') and hasattr(args.client, 'personalization'):
            personalization_config = args.client.personalization
            self.enable_personalization = getattr(personalization_config, "enable", False)
        else:
            self.enable_personalization = False
            
        if hasattr(args, 'cyclical_lr'):
            cyclical_lr_config = args.cyclical_lr
            self.enable_cyclical_lr = getattr(cyclical_lr_config, "enable", False)
        else:
            self.enable_cyclical_lr = False
            
        if hasattr(args, 'client') and hasattr(args.client, 'trust_filtering'):
            trust_config = args.client.trust_filtering
            self.enable_trust_filtering = getattr(trust_config, "enable", False)
        else:
            self.enable_trust_filtering = False
        
        if self.model is not None and self.device is not None:
            self.setup(self.model, self.device)
        
    def setup(self, model, device, optimizer=None):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        if optimizer is None and hasattr(self.args, 'optimizer') and hasattr(self.args.optimizer, 'lr'):
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.optimizer.lr,
                momentum=getattr(self.args.optimizer, 'momentum', 0.9),
                weight_decay=getattr(self.args.optimizer, 'weight_decay', 1e-4)
            )
        else:
            self.optimizer = optimizer
    
    def save_model(self, epoch, suffix=""):
        if not hasattr(self.args, 'checkpoint_path'):
            return
            
        save_path = f"{self.args.checkpoint_path}/model_{epoch}_{suffix}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, save_path)
        logger.info(f"Model saved to {save_path}")
            
    def evaluate(self, epoch):
        self.model.eval()
        
        if isinstance(self.datasets['test'], DataLoader):
            test_loader = self.datasets['test']
        else:
            test_loader = DataLoader(
                self.datasets['test'],
                batch_size=self.args.eval.batch_size,
                shuffle=False,
                num_workers=getattr(self.args, 'num_workers', 0),
                pin_memory=True
            )
        
        global_correct = 0
        personalized_correct = 0
        total = 0
        
        class_correct_global = {}
        class_correct_personalized = {}
        class_total = {}
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.enable_personalization and hasattr(self.model, 'enable_personalized_mode'):
                    self.model.enable_personalized_mode()
                
                output = self.model(data)
                
                if isinstance(output, dict):
                    if "global_logit" in output:
                        global_pred = output["global_logit"].argmax(dim=1)
                        global_correct += global_pred.eq(target).sum().item()
                    elif "logit" in output:
                        global_pred = output["logit"].argmax(dim=1)
                        global_correct += global_pred.eq(target).sum().item()
                    else:
                        global_pred = list(output.values())[0].argmax(dim=1)
                        global_correct += global_pred.eq(target).sum().item()
                    
                    if "personalized_logit" in output:
                        personalized_pred = output["personalized_logit"].argmax(dim=1)
                        personalized_correct += personalized_pred.eq(target).sum().item()
                    elif "logit" in output and self.enable_personalization:
                        personalized_pred = output["logit"].argmax(dim=1)
                        personalized_correct += personalized_pred.eq(target).sum().item()
                    else:
                        personalized_pred = global_pred
                        personalized_correct += personalized_pred.eq(target).sum().item()
                else:
                    pred = output.argmax(dim=1)
                    global_pred = pred
                    personalized_pred = pred
                    global_correct += pred.eq(target).sum().item()
                    personalized_correct += pred.eq(target).sum().item()
                
                for label in torch.unique(target):
                    label_idx = (target == label)
                    label_val = label.item()
                    
                    if label_val not in class_total:
                        class_total[label_val] = 0
                        class_correct_global[label_val] = 0
                        class_correct_personalized[label_val] = 0
                    
                    class_total[label_val] += label_idx.sum().item()
                    class_correct_global[label_val] += ((global_pred == target) & label_idx).sum().item()
                    class_correct_personalized[label_val] += ((personalized_pred == target) & label_idx).sum().item()
                    
                total += target.size(0)
        
        global_acc = 100. * global_correct / total if total > 0 else 0
        personalized_acc = 100. * personalized_correct / total if total > 0 else 0
        
        class_acc_global = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct_global.items()}
        class_acc_personalized = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct_personalized.items()}
        
        balanced_acc_global = sum(class_acc_global.values()) / len(class_acc_global) if class_acc_global else 0
        balanced_acc_personalized = sum(class_acc_personalized.values()) / len(class_acc_personalized) if class_acc_personalized else 0
        
        logger.info(f'Round {epoch} - Global Acc: {global_acc:.2f}%, Personalized Acc: {personalized_acc:.2f}%')
        logger.info(f'Round {epoch} - Balanced Global Acc: {balanced_acc_global:.2f}%, Balanced Personalized Acc: {balanced_acc_personalized:.2f}%')
        
        return {
            'acc': global_acc,
            'acc_personalized': personalized_acc,
            'balanced_acc_global': balanced_acc_global,
            'balanced_acc_personalized': balanced_acc_personalized,
            'class_acc_global': class_acc_global,
            'class_acc_personalized': class_acc_personalized,
            'total': total
        }
    
    def train(self):
        logger.info("Starting federated training...")
        
        if self.server is None or self.client_type is None:
            raise ValueError("Server and client_type must be provided")
            
        self.server.setup(self.model)
        
        num_rounds = self.args.trainer.global_rounds
        num_clients = self.args.trainer.num_clients
        
        if hasattr(self.args.trainer, 'participating_clients'):
            clients_per_round = self.args.trainer.participating_clients
        elif hasattr(self.args.trainer, 'participation_rate'):
            clients_per_round = max(1, int(num_clients * self.args.trainer.participation_rate))
        else:
            clients_per_round = max(1, int(num_clients * 0.1))
            
        logger.info(f"Training with {num_clients} clients, {clients_per_round} per round for {num_rounds} rounds")
        
        clients = {}
        for i in range(num_clients):
            clients[i] = self.client_type(self.args, i, copy.deepcopy(self.model))
        
        best_acc = 0
        best_personalized_acc = 0
            
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Round {round_num}/{num_rounds}")
            
            selected_clients = self.server.select_clients(list(clients.keys()), clients_per_round)
            logger.info(f"Selected clients: {selected_clients}")
            
            global_model = self.server.get_global_model()
            global_state = global_model.state_dict()
            
            client_models = {}
            client_stats = {}
            
            for client_id in selected_clients:
                clients[client_id].setup(
                    state_dict=global_state,
                    device=self.device,
                    local_dataset=self.datasets['train'][client_id] if client_id in self.datasets['train'] else None,
                    global_epoch=round_num,
                    local_lr=self.args.trainer.local_lr,
                    trainer=self
                )
                
                client_state_dict, stats = clients[client_id].local_train(round_num)
                
                if client_state_dict is not None:
                    client_models[client_id] = client_state_dict
                    client_stats[client_id] = stats
            
            if client_models:
                updated_state = self.server.update_global_model(client_models, client_stats=client_stats)
                self.model.load_state_dict(updated_state)
            else:
                logger.warning("No valid client models returned for aggregation")
            
            if round_num % self.args.eval.freq == 0 or round_num == num_rounds:
                eval_results = self.evaluate(round_num)
                logger.info(f"Round {round_num} evaluation: Global Acc = {eval_results['acc']:.2f}%, Personalized Acc = {eval_results['acc_personalized']:.2f}%")
                
                if eval_results['acc'] > best_acc:
                    best_acc = eval_results['acc']
                    self.save_model(round_num, "best_global")
                    
                if eval_results['acc_personalized'] > best_personalized_acc:
                    best_personalized_acc = eval_results['acc_personalized']
                    self.save_model(round_num, "best_personalized")
        
        logger.info(f"Federated training completed. Best global acc: {best_acc:.2f}%, Best personalized acc: {best_personalized_acc:.2f}%")
        return self.model

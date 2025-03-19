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
    """Base trainer class for federated learning experiments.
    
    This class provides the foundation for training models in a federated learning setup,
    with support for personalization, adaptive learning rates, and trust-based filtering.
    """
    
    def __init__(self, args):
        """Initialize the trainer with configuration parameters.
        
        Args:
            args: Configuration parameters for training
        """
        self.args = args
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Personalization settings
        personalization_config = getattr(args, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        
        # Adaptive learning rate settings
        adaptive_lr_config = getattr(args, "adaptive_lr", {})
        self.enable_adaptive_lr = getattr(adaptive_lr_config, "enable", False)
        self.adaptive_lr_beta = getattr(adaptive_lr_config, "beta", 0.1)
        self.adaptive_lr_min = getattr(adaptive_lr_config, "min_lr", 0.001)
        self.adaptive_lr_max = getattr(adaptive_lr_config, "max_lr", 0.1)
        
        # Trust filtering settings
        trust_config = getattr(args, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_config, "enable", False)
        self.trust_threshold = getattr(trust_config, "threshold", 0.5)
        
        # Gradient tracking for adaptive learning rate
        self.grad_history = []
        self.grad_variance = 0.0
        self.previous_model_state = None
        
    def setup(self, model, device, optimizer=None):
        """Set up the trainer with a model and device.
        
        Args:
            model: The model to train
            device: The device to use for training
            optimizer: Optional optimizer (will be created if not provided)
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        if optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optimizer
            
        # Store previous model state for trust score calculation
        if self.enable_trust_filtering:
            self.previous_model_state = copy.deepcopy(model.state_dict())
            
    def train_one_epoch(self, model, train_loader, optimizer, epoch):
        """Train the model for one epoch.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            optimizer: The optimizer to use
            epoch: Current epoch number
            
        Returns:
            float: Average loss for the epoch
        """
        model.train()
        losses = AverageMeter('Loss', ':.4f')
        
        # Calculate gradient variance from previous epoch if available
        if self.enable_adaptive_lr:
            grad_variance = self.calculate_gradient_variance() if len(self.grad_history) > 1 else 0.0
            
            # Adjust learning rate based on gradient variance
            current_lr = self.adjust_learning_rate(optimizer, grad_variance)
            logger.info(f"Adaptive learning rate: {current_lr:.6f} (variance: {grad_variance:.6f})")
        
        # Store gradients for variance calculation
        current_gradients = []
        
        # Use mixed precision training if available
        scaler = GradScaler(enabled=self.device.type == "cuda")
        
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.device.type == "cuda"):
                if hasattr(model, 'forward') and callable(model.forward):
                    output = model(data)
                    
                    # Handle different output formats
                    if isinstance(output, dict):
                        if self.enable_personalization and "personalized_logit" in output:
                            logits = output["personalized_logit"]
                        else:
                            logits = output["logit"] if "logit" in output else output["output"]
                    else:
                        logits = output
                        
                    loss = self.criterion(logits, target)
                else:
                    raise AttributeError("Model must have a callable forward method")
            
            # Backward pass with mixed precision
            if self.device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            
            # Store gradients for adaptive learning rate
            if self.enable_adaptive_lr:
                batch_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        batch_grads.append(param.grad.detach().clone())
                current_gradients.append(batch_grads)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            # Update weights with mixed precision
            if self.device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            losses.update(loss.item(), data.size(0))
            
        # Update gradient history for adaptive learning rate
        if self.enable_adaptive_lr and current_gradients:
            self.grad_history.append(current_gradients)
            # Keep history manageable
            if len(self.grad_history) > 10:
                self.grad_history.pop(0)
        
        end_time = time.time()
        logger.info(f"Epoch {epoch} completed in {end_time - start_time:.2f}s. Loss: {losses.avg:.4f}")
        
        return losses.avg
    
    def calculate_gradient_variance(self):
        """Calculate variance of gradients from previous epochs.
        
        Returns:
            float: Variance of gradients
        """
        if len(self.grad_history) < 2:
            return 0.0
        
        # Calculate average gradient norm per epoch
        epoch_norms = []
        for epoch_grads in self.grad_history:
            batch_norms = []
            for batch_grads in epoch_grads:
                if batch_grads:
                    # Calculate Frobenius norm of all gradients in the batch
                    batch_norm = torch.norm(torch.cat([g.flatten() for g in batch_grads if g is not None]))
                    batch_norms.append(batch_norm.item())
            if batch_norms:
                epoch_norms.append(np.mean(batch_norms))
        
        # Calculate variance of epoch norms
        if len(epoch_norms) > 1:
            return np.var(epoch_norms)
        return 0.0
    
    def adjust_learning_rate(self, optimizer, grad_variance):
        """Adjust learning rate based on gradient variance.
        
        Args:
            optimizer: The optimizer to adjust
            grad_variance: Variance of gradients
            
        Returns:
            float: Adjusted learning rate
        """
        base_lr = self.args.lr
        adjusted_lr = base_lr / (1 + self.adaptive_lr_beta * grad_variance)
        
        # Ensure learning rate stays within bounds
        adjusted_lr = max(min(adjusted_lr, self.adaptive_lr_max), self.adaptive_lr_min)
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr
        
        return adjusted_lr
    
    def compute_trust_score(self, model):
        """Compute trust score for model updates.
        
        Args:
            model: Current model state
            
        Returns:
            float: Trust score between 0 and 1
        """
        if not self.enable_trust_filtering or self.previous_model_state is None:
            return 1.0
        
        # Get current model state
        current_state = model.state_dict()
        
        # Calculate update norm (L2 distance between current and previous model)
        update_norm = 0.0
        param_count = 0
        
        for key in current_state:
            if 'personalized_head' not in key and key in self.previous_model_state:
                diff = current_state[key] - self.previous_model_state[key]
                update_norm += torch.norm(diff).item() ** 2
                param_count += diff.numel()
        
        if param_count > 0:
            update_norm = (update_norm / param_count) ** 0.5
        
        # Trust score is inversely related to update norm
        trust_score = 1.0 / (1.0 + update_norm)
        
        # Update previous model state for next calculation
        self.previous_model_state = copy.deepcopy(current_state)
        
        return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1
    
    def evaluate(self, model, test_loader):
        """Evaluate the model on a test dataset.
        
        Args:
            model: The model to evaluate
            test_loader: DataLoader for test data
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = model(data)
                
                # Handle different output formats
                if isinstance(output, dict):
                    if self.enable_personalization and "personalized_logit" in output:
                        logits = output["personalized_logit"]
                    else:
                        logits = output["logit"] if "logit" in output else output["output"]
                else:
                    logits = output
                
                # Calculate loss
                test_loss += self.criterion(logits, target).item()
                
                # Calculate accuracy
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        
        logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


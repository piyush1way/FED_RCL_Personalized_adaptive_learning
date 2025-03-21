import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'get_numclasses', 
    'count_label_distribution', 
    'check_data_distribution', 
    'get_optimizer', 
    'get_scheduler',
    'modeleval', 
    'create_pth_dict',
    'save_dict_to_json',
    'load_dict_from_json',
    'compute_trust_score',
    'setup_adaptive_learning_rate',
    'get_prefix_num'  # Added missing function referenced in create_pth_dict
]

def count_label_distribution(labels, class_num:int=10, default_dist:torch.tensor=None):
    if default_dist is not None:
        default = default_dist
    else:
        default = torch.zeros(class_num)
    data_distribution = default
    for idx, label in enumerate(labels):
        data_distribution[label] += 1 
    data_distribution = data_distribution / data_distribution.sum()
    return data_distribution

def check_data_distribution(dataloader, class_num:int=10, default_dist:torch.tensor=None):
    if default_dist is not None:
        default = default_dist
    else:
        default = torch.zeros(class_num)
    data_distribution = default
    for idx, (images, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i] += 1 
    data_distribution = data_distribution / data_distribution.sum()
    return data_distribution

def get_numclasses(args, trainset=None):
    if args.dataset.name in ['CIFAR10', "MNIST"]:
        num_classes = 10
    elif args.dataset.name in ["CIFAR100"]:
        num_classes = 100
    elif args.dataset.name in ["TinyImageNet"]:
        num_classes = 200
    elif args.dataset.name in ["iNaturalist"]:
        num_classes = 1203
    elif args.dataset.name in ["ImageNet"]:
        num_classes = 1000
    elif args.dataset.name in ["leaf_celeba"]:
        num_classes = 2
    elif args.dataset.name in ["leaf_femnist"]:
        num_classes = 62
    elif args.dataset.name in ["shakespeare"]:
        num_classes = 80
    else:
        logger.error(f"Unknown dataset: {args.dataset.name}")
        raise ValueError(f"Unknown dataset: {args.dataset.name}")
        
    logger.info(f"Number of classes for {args.dataset.name}: {num_classes}")
    return num_classes

def get_optimizer(args, parameters):
    if args.optimizer.name.lower() == 'sgd':
        optimizer = optim.SGD(
            parameters, 
            lr=args.optimizer.get('lr', 0.01),
            momentum=args.optimizer.get('momentum', 0.9), 
            weight_decay=args.optimizer.get('wd', 1e-4)
        )
    elif args.optimizer.name.lower() == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=args.optimizer.get('lr', 0.001),
            weight_decay=args.optimizer.get('wd', 1e-4)
        )
    else:
        logger.error(f"Invalid optimizer: {args.optimizer.name}")
        raise ValueError(f"Invalid optimizer: {args.optimizer.name}")
    
    return optimizer

def get_scheduler(optimizer, args):
    if not hasattr(args, 'scheduler') or args.scheduler.name.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=args.scheduler.get('gamma', 0.998)
        )
    elif args.scheduler.name.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler.get('step_size', 30),
            gamma=args.scheduler.get('gamma', 0.1)
        )
    elif args.scheduler.name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler.get('T_max', 200)
        )
    else:
        logger.error(f"Invalid scheduler: {args.scheduler.name}")
        raise ValueError(f"Invalid scheduler: {args.scheduler.name}")
        
    return scheduler

def modeleval(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                outputs = outputs.get("logit", outputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / float(total) if total > 0 else 0
    logger.info(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    model.train()
    return accuracy

def get_prefix_num(filename):
    """Extract prefix and number from a filename."""
    parts = filename.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        prefix = '_'.join(parts[:-1])
        number = int(parts[-1])
        return prefix, number
    else:
        return filename, 0

def create_pth_dict(pth_path):
    pth_dir = os.path.dirname(pth_path)
    pth_base = os.path.basename(pth_path)
    pth_prefix, _ = get_prefix_num(pth_base)

    pth_dict = {}

    for filename in os.listdir(pth_dir):
        if filename.startswith(pth_prefix):
            _, number = get_prefix_num(filename)
            filepath = os.path.join(pth_dir, filename)
            pth_dict[number] = filepath

    return dict(sorted(pth_dict.items()))

def save_dict_to_json(d, json_path):
    """Save a dictionary to a JSON file.
    
    Args:
        d: Dictionary to save
        json_path: Path to the JSON file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(json_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    # Convert any non-serializable objects (like numpy arrays or tensors)
    serializable_dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            serializable_dict[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            serializable_dict[k] = v.cpu().tolist()
        elif isinstance(v, dict):
            # Handle nested dictionaries
            nested_dict = {}
            for nk, nv in v.items():
                if isinstance(nv, np.ndarray):
                    nested_dict[nk] = nv.tolist()
                elif isinstance(nv, torch.Tensor):
                    nested_dict[nk] = nv.cpu().tolist()
                else:
                    nested_dict[nk] = nv
            serializable_dict[k] = nested_dict
        else:
            serializable_dict[k] = v
    
    # Save to JSON file
    with open(json_path, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
        
    logger.info(f"Dictionary saved to {json_path}")

def load_dict_from_json(json_path):
    """Load a dictionary from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary loaded from the JSON file
    """
    if not os.path.exists(json_path):
        logger.warning(f"JSON file not found: {json_path}")
        return {}
        
    with open(json_path, 'r') as f:
        d = json.load(f)
        
    logger.info(f"Dictionary loaded from {json_path}")
    return d

def compute_trust_score(current_model, previous_model, update_history=None):
    """Compute trust score based on model update magnitude and consistency.
    
    Args:
        current_model: Current model state
        previous_model: Previous model state
        update_history: List of previous update magnitudes
        
    Returns:
        Trust score between 0 and 1
    """
    if previous_model is None:
        return 1.0
        
    # Calculate update magnitude
    update_norm = 0.0
    param_count = 0
    
    for key in current_model:
        if 'personalized_head' not in key and key in previous_model:
            current_param = current_model[key].float()
            prev_param = previous_model[key].float()
            diff = current_param - prev_param
            update_norm += torch.norm(diff).item() ** 2
            param_count += diff.numel()
    
    if param_count > 0:
        update_norm = (update_norm / param_count) ** 0.5
    
    # Calculate update consistency if history is provided
    if update_history is not None:
        update_history.append(update_norm)
        if len(update_history) > 5:
            update_history.pop(0)
            
        update_variance = np.var(update_history) if len(update_history) > 1 else 0.0
        
        # Compute trust score components
        magnitude_score = 1.0 / (1.0 + update_norm)
        consistency_score = 1.0 / (1.0 + update_variance)
        
        # Combine scores with weights
        trust_score = 0.7 * magnitude_score + 0.3 * consistency_score
    else:
        # Simpler score based only on magnitude if no history
        trust_score = 1.0 / (1.0 + update_norm)
    
    return max(0.0, min(1.0, trust_score))

def setup_adaptive_learning_rate(base_lr, max_lr, trust_score, step, step_size):
    """Set up adaptive learning rate based on trust score and cyclical schedule.
    
    Args:
        base_lr: Base learning rate
        max_lr: Maximum learning rate
        trust_score: Client trust score
        step: Current step in the cycle
        step_size: Number of steps in half a cycle
        
    Returns:
        Adjusted learning rate
    """
    # Cyclical LR calculation
    cycle_step = step % (2 * step_size)
    x = abs(cycle_step / step_size - 1)
    
    # Adjust max_lr based on trust score
    adjusted_max_lr = max_lr * (0.5 + 0.5 * trust_score)
    
    # Calculate final learning rate
    adjusted_lr = base_lr + (adjusted_max_lr - base_lr) * max(0, (1 - x))
    
    return adjusted_lr

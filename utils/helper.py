import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import json
import numpy as np
import logging
import math

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

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_dict_to_json(d: dict, json_path: str):
    """Save dict to json file
    
    Args:
        d: dict to save
        json_path: path to json file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Convert dict to JSON serializable format
    serializable_dict = convert_to_serializable(d)
    
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

def setup_adaptive_learning_rate(min_lr, max_lr, current_round, total_rounds, trust_weight=1.0, client_id=0):
    """
    Set up adaptive learning rate based on current round and trust weight.
    Uses cyclical learning rate pattern with trust-based scaling.
    
    Args:
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        current_round: Current training round
        total_rounds: Total number of training rounds
        trust_weight: Trust score to scale learning rate (higher trust = higher LR)
        client_id: Client ID to ensure variation across clients
        
    Returns:
        Calculated learning rate
    """
    # Normalize current position in training
    position = (current_round % (total_rounds // 5)) / (total_rounds // 5)
    
    # Add client-specific variation
    client_factor = 0.8 + (client_id % 10) * 0.04  # Varies from 0.8 to 1.16
    position = (position + client_id * 0.01) % 1.0  # Shifts the cycle position
    
    # Convert position to angle
    angle = position * 2 * math.pi
    
    # Create cosine wave scaled between 0 and 1
    cosine_value = (math.cos(angle) + 1) / 2
    
    # Calculate base learning rate (cosine annealing)
    base_lr = min_lr + (max_lr - min_lr) * cosine_value
    
    # Scale by trust weight (higher trust = higher learning rate)
    # Add client-specific scaling
    trust_adjusted_lr = base_lr * (0.5 + 0.5 * trust_weight) * client_factor
    
    return trust_adjusted_lr

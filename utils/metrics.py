import copy
import torch
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from scipy import stats
import time
import torch.nn.functional as F
from utils.meter import AverageMeter

logger = logging.getLogger(__name__)

__all__ = ['evaluate', 'track_trust_scores', 'evaluate_personalized_clients', 'evaluate_personalization_benefits']

def evaluate(args, model, test_loader, device, criterion=None):
    """Evaluate the model on the test dataset"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # For computing balanced accuracy
    true_positives = {}
    class_counts = {}
    
    # For tracking confidence
    confidences = []
    
    # Track test running time
    batch_time = AverageMeter('Batch Time', ':6.3f')
    start_time = time.time()
    
    # Set max samples to evaluate to avoid long eval time
    max_samples = getattr(args.trainer.eval, 'max_samples', 10000)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Limit number of samples for faster evaluation
            if total >= max_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle both dictionary and direct tensor outputs
            if isinstance(outputs, dict):
                outputs = outputs.get('logit', outputs.get('global_logit', None))
                if outputs is None:
                    for key in ['output', 'pred', 'prediction', 'logits']:
                        if key in outputs:
                            outputs = outputs[key]
                            break
                    if outputs is None:
                        # Last attempt: just use the first value that's a tensor
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor) and value.dim() > 1:
                                outputs = value
                                break
                        
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # Update balanced accuracy metrics
            for cls in torch.unique(targets):
                cls_idx = cls.item()
                if cls_idx not in true_positives:
                    true_positives[cls_idx] = 0
                    class_counts[cls_idx] = 0
                
                cls_mask = (targets == cls_idx)
                true_positives[cls_idx] += predicted[cls_mask].eq(targets[cls_mask]).sum().item()
                class_counts[cls_idx] += cls_mask.sum().item()
            
            # Collect confidence scores
            with torch.no_grad():
                probabilities = F.softmax(outputs, dim=1)
                top_probs, _ = torch.max(probabilities, dim=1)
                confidences.extend(top_probs.cpu().tolist())
            
            # Update timer
            batch_time.update(time.time() - start_time)
            start_time = time.time()
    
    # Calculate final metrics
    acc_global = 100.0 * correct / total if total > 0 else 0.0
    
    # Compute balanced accuracy (average of per-class accuracies)
    balanced_acc = 0.0
    valid_classes = 0
    per_class_acc = {}
    
    for cls in true_positives.keys():
        if class_counts[cls] > 0:
            cls_acc = 100.0 * true_positives[cls] / class_counts[cls]
            per_class_acc[cls] = cls_acc
            balanced_acc += cls_acc
            valid_classes += 1
    
    # Calculate average
    balanced_acc = balanced_acc / valid_classes if valid_classes > 0 else 0.0
    
    # Calculate average loss if criterion was provided
    avg_loss = test_loss / len(test_loader) if criterion is not None else None
    
    # Calculate confidence metrics
    avg_confidence = np.mean(confidences) if confidences else 0.0
    confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
    
    # Log results - fix the f-string format issue
    if criterion is not None:
        logger.info(f'Global Model Accuracy: {acc_global:.2f}% | Loss: {avg_loss:.4f}')
    else:
        logger.info(f'Global Model Accuracy: {acc_global:.2f}%')
        
    # Return all metrics in a dictionary
    metrics = {
        'acc': acc_global / 100.0,  # Return as fraction for easier processing
        'balanced_acc': balanced_acc / 100.0,
        'per_class_acc': {k: v / 100.0 for k, v in per_class_acc.items()},
        'avg_confidence': avg_confidence,
        'confidence_var': confidence_variance,
        'evaluated_samples': total
    }
    
    # Add loss to metrics if calculated
    if avg_loss is not None:
        metrics['loss'] = avg_loss
        
    return metrics


def evaluate_personalized_clients(trainer, testloader, device, num_clients=None):
    """Evaluate personalized models for each client.
    
    Args:
        trainer: The trainer object containing clients
        testloader: DataLoader for test data
        device: Device to run evaluation on
        num_clients: Number of clients to evaluate (None for all)
        
    Returns:
        dict: Per-client evaluation metrics
    """
    if not hasattr(trainer, 'clients'):
        logger.warning("Trainer does not have clients to evaluate")
        return {}
    
    clients = trainer.clients
    if num_clients is not None:
        client_ids = list(clients.keys())[:num_clients]
    else:
        client_ids = list(clients.keys())
    
    results = {}
    
    for client_id in client_ids:
        logger.info(f"Evaluating personalized model for client {client_id}")
        client = clients[client_id]
        
        # Ensure client model is in evaluation mode and on the correct device
        client.model.eval()
        client.model.to(device)
        
        # Enable personalized mode if available
        if hasattr(client.model, 'enable_personalized_mode'):
            client.model.enable_personalized_mode()
        elif hasattr(client.model, 'use_personalized_head'):
            client.model.use_personalized_head = True
        
        # Initialize metrics
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = client.model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get("personalized_logit", outputs.get("logit", list(outputs.values())[0]))
                else:
                    logits = outputs
                
                # Calculate predictions
                _, predicted = torch.max(logits, 1)
                
                # Update counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update class-wise accuracy
                for label in torch.unique(labels):
                    label_idx = (labels == label)
                    label_val = label.item()
                    
                    class_total[label_val] += label_idx.sum().item()
                    class_correct[label_val] += ((predicted == labels) & label_idx).sum().item()
        
        # Calculate accuracy
        acc = 100 * correct / total if total > 0 else 0
        
        # Calculate class-wise accuracy
        class_acc = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct.items() if class_total[cls] > 0}
        
        # Calculate balanced accuracy
        balanced_acc = sum(class_acc.values()) / len(class_acc) if class_acc else 0
        
        # Store results
        results[client_id] = {
            "acc": acc,
            "balanced_acc": balanced_acc,
            "class_acc": class_acc
        }
        
        logger.info(f"Client {client_id} - Accuracy: {acc:.2f}%, Balanced Accuracy: {balanced_acc:.2f}%")
        
        # Move model back to CPU to save memory
        client.model.to('cpu')
    
    return results


def track_trust_scores(trainer, num_clients=None):
    """Track client trust scores and evaluate per-client personalized models.
    
    Args:
        trainer: The trainer object containing the server with trust scores
        num_clients: Number of clients to evaluate (optional)
        
    Returns:
        dict: Statistics about trust scores and per-client performance
    """
    if not hasattr(trainer.server, 'trust_scores'):
        logger.warning("Server does not have trust scores to track")
        return {}
    
    trust_scores = trainer.server.trust_scores
    log_dir = str(trainer.args.log_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Save trust scores
    with open(os.path.join(log_dir, 'final_trust_scores.json'), 'w') as f:
        json.dump({str(k): float(v) for k, v in trust_scores.items()}, f)
    
    # Calculate statistics
    scores = list(trust_scores.values())
    stats = {
        'mean_trust': np.mean(scores) if scores else 0,
        'median_trust': np.median(scores) if scores else 0,
        'min_trust': min(scores) if scores else 0,
        'max_trust': max(scores) if scores else 0,
        'std_trust': np.std(scores) if scores else 0,
        'trusted_clients': sum(1 for s in scores if s >= 0.5),
        'total_clients': len(scores)
    }
    
    # Generate visualization
    try:
        plt.figure(figsize=(10, 6))
        client_ids = sorted(trust_scores.keys())
        client_scores = [trust_scores[cid] for cid in client_ids]
        
        plt.bar(range(len(client_ids)), client_scores)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Trust Threshold')
        plt.xlabel('Client ID')
        plt.ylabel('Trust Score')
        plt.title('Client Trust Scores')
        plt.xticks(range(len(client_ids)), client_ids)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'final_trust_scores.png'))
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate trust score visualization: {e}")
    
    return stats


def evaluate_personalization_benefits(args, model, test_loader, device, criterion=None):
    """Evaluate the benefits of personalization compared to global model"""
    if not (hasattr(model, 'enable_personalized_mode') and hasattr(model, 'disable_personalized_mode')):
        return {
            "bop": 0.0,
            "global_acc": 0.0,
            "personalized_acc": 0.0,
            "acc_global": 0.0,
            "acc_personalized": 0.0
        }
    
    # Setup
    model.eval()
    global_loss, personalized_loss = 0.0, 0.0
    global_correct, personalized_correct = 0, 0
    global_preds, personalized_preds, all_labels = [], [], []
    total = 0
    
    # Limit samples to avoid OOM
    max_samples = getattr(args.trainer.eval, 'max_samples', 500)
    
    # Evaluate global model first
    model.disable_personalized_mode()
    
    # Prepare class accuracy tracking
    class_correct_global = defaultdict(int)
    class_correct_personalized = defaultdict(int)
    class_total = defaultdict(int)
    
    # Track confidences
    global_confs, personalized_confs = [], []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Limit evaluation samples
            if total >= max_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Global model evaluation
            model.disable_personalized_mode()
            outputs_global = model(inputs)
            if isinstance(outputs_global, dict):
                outputs_global = outputs_global.get('logit', outputs_global.get('global_logit', list(outputs_global.values())[0]))
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs_global, targets)
                global_loss += loss.item()
                
            # Global model predictions
            global_probs = F.softmax(outputs_global, dim=1)
            global_conf, global_pred = torch.max(global_probs, 1)
            global_correct += global_pred.eq(targets).sum().item()
            global_confs.extend(global_conf.cpu().numpy())
            global_preds.extend(global_pred.cpu().numpy())
            
            # Personalized model evaluation
            model.enable_personalized_mode()
            outputs_personalized = model(inputs)
            if isinstance(outputs_personalized, dict):
                outputs_personalized = outputs_personalized.get('logit', outputs_personalized.get('personalized_logit', list(outputs_personalized.values())[0]))
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs_personalized, targets)
                personalized_loss += loss.item()
                
            # Personalized model predictions
            pers_probs = F.softmax(outputs_personalized, dim=1)
            pers_conf, pers_pred = torch.max(pers_probs, 1)
            personalized_correct += pers_pred.eq(targets).sum().item()
            personalized_confs.extend(pers_conf.cpu().numpy())
            personalized_preds.extend(pers_pred.cpu().numpy())
            
            # Track labels for later analysis
            all_labels.extend(targets.cpu().numpy())
            total += targets.size(0)
            
            # Track class-wise accuracy
            for label in torch.unique(targets):
                label_idx = (targets == label)
                label_val = label.item()
                
                class_total[label_val] += label_idx.sum().item()
                class_correct_global[label_val] += ((global_pred == targets) & label_idx).sum().item()
                class_correct_personalized[label_val] += ((pers_pred == targets) & label_idx).sum().item()
    
    # Calculate final metrics
    global_acc = global_correct / total if total > 0 else 0.0
    personalized_acc = personalized_correct / total if total > 0 else 0.0
    
    # Calculate average loss if criterion was provided
    avg_global_loss = global_loss / len(test_loader) if criterion is not None else None
    avg_personalized_loss = personalized_loss / len(test_loader) if criterion is not None else None
    
    # Calculate per-class accuracies and benefit of personalization
    class_bop = {}
    for cls in class_total:
        if class_total[cls] > 0:
            global_cls_acc = class_correct_global[cls] / class_total[cls]
            pers_cls_acc = class_correct_personalized[cls] / class_total[cls]
            class_bop[cls] = max(0, pers_cls_acc - global_cls_acc)  # BOP can't be negative
    
    # Benefit of personalization (improvement in accuracy)
    bop = max(0, personalized_acc - global_acc)
    
    # Log results with proper formatting
    if criterion is not None:
        logger.info(f'Global Model Acc: {global_acc*100:.2f}% | Loss: {avg_global_loss:.4f}')
        logger.info(f'Personalized Model Acc: {personalized_acc*100:.2f}% | Loss: {avg_personalized_loss:.4f}')
    else:
        logger.info(f'Global Model Acc: {global_acc*100:.2f}%')
        logger.info(f'Personalized Model Acc: {personalized_acc*100:.2f}%')
    
    logger.info(f'Benefit of Personalization: {bop*100:.2f}%')
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return {
        "bop": bop,
        "global_acc": global_acc,
        "personalized_acc": personalized_acc,
        "acc_global": global_acc,
        "acc_personalized": personalized_acc,
        "class_bop": class_bop,
        "global_loss": avg_global_loss if criterion is not None else None,
        "personalized_loss": avg_personalized_loss if criterion is not None else None
    }

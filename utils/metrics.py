import copy
import torch
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = ['evaluate', 'track_trust_scores', 'evaluate_personalized_clients', 'evaluate_personalization_benefits']

def evaluate(args, model, testloader, device) -> dict:
    """
    Evaluate both global and personalized models on the test set.
    
    Args:
        args: Configuration arguments
        model: The model to evaluate
        testloader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics including accuracy and class-wise performance
    """
    eval_device = device if not args.multiprocessing else torch.device(f"cuda:{args.main_gpu}")
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    eval_model.to(eval_device)

    # Initialize counters
    correct_global, correct_personalized, total = 0, 0, 0
    
    # Check if model supports personalization
    has_personalization = hasattr(eval_model, 'personalized_head') or hasattr(eval_model, 'use_personalized_head')
    
    # Track class-wise accuracy for both models
    class_correct_global = defaultdict(int)
    class_correct_personalized = defaultdict(int)
    class_total = defaultdict(int)
    
    # Track confidence scores
    confidence_global = []
    confidence_personalized = []
    
    # Track per-sample predictions for analysis
    all_labels = []
    all_global_preds = []
    all_personalized_preds = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(eval_device), labels.to(eval_device)
            
            # Forward pass
            outputs = eval_model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if "global_logit" in outputs and "personalized_logit" in outputs:
                    # Model returns both global and personalized logits
                    global_logits = outputs["global_logit"]
                    personalized_logits = outputs["personalized_logit"]
                elif "logit" in outputs and has_personalization:
                    # Model returns only one set of logits based on current mode
                    # Store current personalization mode
                    current_mode = getattr(eval_model, 'use_personalized_head', False)
                    
                    # Get global logits
                    if hasattr(eval_model, 'use_personalized_head'):
                        eval_model.use_personalized_head = False
                    global_logits = eval_model(images)
                    if isinstance(global_logits, dict):
                        global_logits = global_logits["logit"]
                    
                    # Get personalized logits
                    if hasattr(eval_model, 'use_personalized_head'):
                        eval_model.use_personalized_head = True
                    personalized_logits = eval_model(images)
                    if isinstance(personalized_logits, dict):
                        personalized_logits = personalized_logits["logit"]
                    
                    # Restore original mode
                    if hasattr(eval_model, 'use_personalized_head'):
                        eval_model.use_personalized_head = current_mode
                else:
                    # Only global model available
                    global_logits = outputs["logit"] if "logit" in outputs else outputs.get("output", list(outputs.values())[0])
                    personalized_logits = global_logits  # No personalization
            else:
                # Model returns tensor directly
                global_logits = outputs
                personalized_logits = outputs  # No personalization
            
            # Calculate predictions
            global_probs = torch.softmax(global_logits, dim=1)
            personalized_probs = torch.softmax(personalized_logits, dim=1)
            
            global_conf, predicted_global = torch.max(global_probs, 1)
            personalized_conf, predicted_personalized = torch.max(personalized_probs, 1)
            
            # Store confidence scores
            confidence_global.extend(global_conf.cpu().tolist())
            confidence_personalized.extend(personalized_conf.cpu().tolist())
            
            # Store predictions for analysis
            all_labels.extend(labels.cpu().tolist())
            all_global_preds.extend(predicted_global.cpu().tolist())
            all_personalized_preds.extend(predicted_personalized.cpu().tolist())
            
            # Update counters
            total += labels.size(0)
            correct_global += (predicted_global == labels).sum().item()
            correct_personalized += (predicted_personalized == labels).sum().item()
            
            # Update class-wise accuracy
            for label in torch.unique(labels):
                label_idx = (labels == label)
                label_val = label.item()
                
                class_total[label_val] += label_idx.sum().item()
                class_correct_global[label_val] += ((predicted_global == labels) & label_idx).sum().item()
                class_correct_personalized[label_val] += ((predicted_personalized == labels) & label_idx).sum().item()

    # Calculate overall accuracy
    acc_global = 100 * correct_global / float(total) if total > 0 else 0
    acc_personalized = 100 * correct_personalized / float(total) if total > 0 else 0
    
    # Calculate class-wise accuracy
    class_acc_global = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct_global.items()}
    class_acc_personalized = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct_personalized.items()}
    
    # Calculate balanced accuracy (average of class-wise accuracies)
    balanced_acc_global = sum(class_acc_global.values()) / len(class_acc_global) if class_acc_global else 0
    balanced_acc_personalized = sum(class_acc_personalized.values()) / len(class_acc_personalized) if class_acc_personalized else 0
    
    # Calculate average confidence
    avg_confidence_global = sum(confidence_global) / len(confidence_global) if confidence_global else 0
    avg_confidence_personalized = sum(confidence_personalized) / len(confidence_personalized) if confidence_personalized else 0

    logger.info(f'Global Model Accuracy: {acc_global:.2f}% | Personalized Model Accuracy: {acc_personalized:.2f}%')
    logger.info(f'Global Model Balanced Accuracy: {balanced_acc_global:.2f}% | Personalized Model Balanced Accuracy: {balanced_acc_personalized:.2f}%')

    # Clean up
    eval_model.to('cpu')
    torch.cuda.empty_cache()
    
    return {
        "acc": acc_global,  # Keep original key for compatibility
        "acc_personalized": acc_personalized,
        "balanced_acc_global": balanced_acc_global,
        "balanced_acc_personalized": balanced_acc_personalized,
        "class_acc_global": class_acc_global,
        "class_acc_personalized": class_acc_personalized,
        "avg_confidence_global": avg_confidence_global,
        "avg_confidence_personalized": avg_confidence_personalized,
        "predictions": {
            "labels": all_labels,
            "global_preds": all_global_preds,
            "personalized_preds": all_personalized_preds
        }
    }


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


def evaluate_personalization_benefits(args, model, testloader, device, client_data_distributions=None):
    """
    Evaluate the benefits of personalization in federated learning scenarios.
    
    Args:
        args: Configuration arguments
        model: The model to evaluate
        testloader: DataLoader for test data
        device: Device to run evaluation on
        client_data_distributions: Dict mapping client IDs to data distribution info
        
    Returns:
        dict: Metrics quantifying personalization benefits
    """
    # Create a copy of the model for evaluation
    eval_device = device if not args.multiprocessing else torch.device(f"cuda:{args.main_gpu}")
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    eval_model.to(eval_device)
    
    # Check if model supports personalization
    if not (hasattr(eval_model, 'personalized_head') or hasattr(eval_model, 'use_personalized_head')):
        logger.warning("Model does not support personalization")
        return {"bop": 0, "has_personalization": False}
    
    # Initialize metrics
    global_preds = []
    personalized_preds = []
    true_labels = []
    
    # Per-class and per-sample tracking
    class_improvement = defaultdict(list)
    class_counts = defaultdict(int)
    sample_confidences = {'global': [], 'personalized': []}
    
    # Evaluate both models on test data
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(eval_device), labels.to(eval_device)
            
            # Get global model predictions
            if hasattr(eval_model, 'use_personalized_head'):
                eval_model.use_personalized_head = False
            global_output = eval_model(images)
            if isinstance(global_output, dict):
                global_logits = global_output.get("global_logit", global_output.get("logit", None))
            else:
                global_logits = global_output
                
            # Get personalized model predictions
            if hasattr(eval_model, 'use_personalized_head'):
                eval_model.use_personalized_head = True
            personalized_output = eval_model(images)
            if isinstance(personalized_output, dict):
                personalized_logits = personalized_output.get("personalized_logit", personalized_output.get("logit", None))
            else:
                personalized_logits = personalized_output
            
            # Calculate predictions and confidence scores
            global_probs = torch.softmax(global_logits, dim=1)
            personalized_probs = torch.softmax(personalized_logits, dim=1)
            
            _, global_pred = torch.max(global_probs, 1)
            _, personalized_pred = torch.max(personalized_probs, 1)
            
            # Calculate confidence scores
            global_conf = torch.gather(global_probs, 1, global_pred.unsqueeze(1)).squeeze(1)
            personalized_conf = torch.gather(personalized_probs, 1, personalized_pred.unsqueeze(1)).squeeze(1)
            
            # Store predictions and labels for overall metrics
            global_preds.extend(global_pred.cpu().tolist())
            personalized_preds.extend(personalized_pred.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
            
            # Per-class improvement
            for i, label in enumerate(labels):
                label_val = label.item()
                class_counts[label_val] += 1
                
                # Record improvement or regression
                global_correct = (global_pred[i] == label).item()
                personalized_correct = (personalized_pred[i] == label).item()
                
                # 1 if personalized is better, -1 if worse, 0 if same
                improvement = personalized_correct - global_correct
                class_improvement[label_val].append(improvement)
                
                # Record confidence scores
                sample_confidences['global'].append(global_conf[i].item())
                sample_confidences['personalized'].append(personalized_conf[i].item())
    
    # Calculate overall accuracy
    global_acc = sum(1 for y_true, y_pred in zip(true_labels, global_preds) if y_true == y_pred) / len(true_labels) if true_labels else 0
    personalized_acc = sum(1 for y_true, y_pred in zip(true_labels, personalized_preds) if y_true == y_pred) / len(true_labels) if true_labels else 0
    
    # Calculate Benefit of Personalization (BoP)
    bop = personalized_acc - global_acc
    
    # Calculate per-class BoP
    class_bop = {}
    for cls, improvements in class_improvement.items():
        class_bop[cls] = sum(improvements) / len(improvements)
    
    # Calculate minimum group BoP (BoP for the worst-performing group)
    min_class_bop = min(class_bop.values()) if class_bop else 0
    
    # Calculate confidence-based metrics
    avg_conf_global = sum(sample_confidences['global']) / len(sample_confidences['global']) if sample_confidences['global'] else 0
    avg_conf_personalized = sum(sample_confidences['personalized']) / len(sample_confidences['personalized']) if sample_confidences['personalized'] else 0
    
    # Calculate confusion matrices
    num_classes = max(max(true_labels) + 1 if true_labels else 0, 
                     args.model.num_classes if hasattr(args.model, 'num_classes') else 10)
    confusion_global = confusion_matrix(true_labels, global_preds, 
                                       labels=list(range(num_classes))).tolist() if true_labels else []
    confusion_personalized = confusion_matrix(true_labels, personalized_preds, 
                                            labels=list(range(num_classes))).tolist() if true_labels else []
    
    # Calculate statistical significance of improvement
    # Using McNemar's test for paired nominal data
    # b: cases where global correct, personalized wrong
    # c: cases where global wrong, personalized correct
    b = sum(1 for g, p, t in zip(global_preds, personalized_preds, true_labels) if g == t and p != t)
    c = sum(1 for g, p, t in zip(global_preds, personalized_preds, true_labels) if g != t and p == t)
    
    # McNemar's test
    if b + c > 0:
        p_value = stats.binom_test(min(b, c), b + c, p=0.5)
        is_significant = p_value < 0.05
    else:
        p_value = 1.0
        is_significant = False
    
    # Compile and return results
    results = {
        "bop": bop,
        "acc_global": global_acc,
        "acc_personalized": personalized_acc,
        "class_bop": class_bop,
        "min_class_bop": min_class_bop,
        "avg_confidence_global": avg_conf_global,
        "avg_confidence_personalized": avg_conf_personalized,
        "confusion_matrix_global": confusion_global,
        "confusion_matrix_personalized": confusion_personalized,
        "statistical_significance": {
            "p_value": p_value,
            "is_significant": is_significant,
            "b": b,  # global correct, personalized wrong
            "c": c   # global wrong, personalized correct
        },
        "has_personalization": True
    }
    
    # Log key results
    logger.info(f"Benefit of Personalization (BoP): {bop:.4f} (p={p_value:.4f}, significant: {is_significant})")
    logger.info(f"Global Model Accuracy: {global_acc:.4f}, Personalized Model Accuracy: {personalized_acc:.4f}")
    logger.info(f"Min Class BoP: {min_class_bop:.4f} (worst performing class)")
    
    # Clean up
    eval_model.to('cpu')
    torch.cuda.empty_cache()
    
    return results

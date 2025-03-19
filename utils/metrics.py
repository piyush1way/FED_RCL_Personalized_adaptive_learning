# import copy
# import torch
# import numpy as np
# from torchmetrics import Metric
# import logging

# logger = logging.getLogger(__name__)

# __all__ = ['evaluate', 'track_trust_scores']

# class AccumTensor(Metric):
#     def __init__(self, default_value: torch.Tensor):
#         super().__init__()
#         self.add_state("val", default=default_value, dist_reduce_fx="sum")

#     def update(self, input_tensor: torch.Tensor):
#         self.val += input_tensor

#     def compute(self):
#         return self.val


# def evaluate(args, model, testloader, device) -> dict:
#     """
#     Evaluate both global and personalized models on the test set.

#     Args:
#         args: Configuration arguments.
#         model: Trained model.
#         testloader: DataLoader for the test dataset.
#         device: Device for evaluation.

#     Returns:
#         dict: Accuracy of both global and personalized models.
#     """
#     eval_device = device if not args.multiprocessing else torch.device(f"cuda:{args.main_gpu}")
#     eval_model = copy.deepcopy(model)
#     eval_model.eval()
#     eval_model.to(eval_device)

#     # Initialize counters
#     correct_global, correct_personalized, total = 0, 0, 0
    
#     # Check if model supports personalization
#     has_personalization = hasattr(eval_model, 'personalized_head') or hasattr(eval_model, 'use_personalized_head')
    
#     # Track class-wise accuracy for both models
#     class_correct_global = {}
#     class_correct_personalized = {}
#     class_total = {}

#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(eval_device), labels.to(eval_device)
            
#             # Forward pass
#             outputs = eval_model(images)
            
#             # Handle different output formats
#             if isinstance(outputs, dict):
#                 if "global_logit" in outputs and "personalized_logit" in outputs:
#                     # Model returns both global and personalized logits
#                     global_logits = outputs["global_logit"]
#                     personalized_logits = outputs["personalized_logit"]
#                 elif "logit" in outputs and has_personalization:
#                     # Model returns only one set of logits based on current mode
#                     # Store current personalization mode
#                     current_mode = getattr(eval_model, 'use_personalized_head', False)
                    
#                     # Get global logits
#                     if hasattr(eval_model, 'use_personalized_head'):
#                         eval_model.use_personalized_head = False
#                     global_logits = eval_model(images)
#                     if isinstance(global_logits, dict):
#                         global_logits = global_logits["logit"]
                    
#                     # Get personalized logits
#                     if hasattr(eval_model, 'use_personalized_head'):
#                         eval_model.use_personalized_head = True
#                     personalized_logits = eval_model(images)
#                     if isinstance(personalized_logits, dict):
#                         personalized_logits = personalized_logits["logit"]
                    
#                     # Restore original mode
#                     if hasattr(eval_model, 'use_personalized_head'):
#                         eval_model.use_personalized_head = current_mode
#                 else:
#                     # Only global model available
#                     global_logits = outputs["logit"] if "logit" in outputs else outputs["output"]
#                     personalized_logits = global_logits  # No personalization
#             else:
#                 # Model returns tensor directly
#                 global_logits = outputs
#                 personalized_logits = outputs  # No personalization
            
#             # Calculate predictions
#             _, predicted_global = torch.max(global_logits, 1)
#             _, predicted_personalized = torch.max(personalized_logits, 1)
            
#             # Update counters
#             total += labels.size(0)
#             correct_global += (predicted_global == labels).sum().item()
#             correct_personalized += (predicted_personalized == labels).sum().item()
            
#             # Update class-wise accuracy
#             for label in torch.unique(labels):
#                 label_idx = (labels == label)
#                 label_val = label.item()
                
#                 if label_val not in class_total:
#                     class_total[label_val] = 0
#                     class_correct_global[label_val] = 0
#                     class_correct_personalized[label_val] = 0
                
#                 class_total[label_val] += label_idx.sum().item()
#                 class_correct_global[label_val] += ((predicted_global == labels) & label_idx).sum().item()
#                 class_correct_personalized[label_val] += ((predicted_personalized == labels) & label_idx).sum().item()

#     # Calculate overall accuracy
#     acc_global = 100 * correct_global / float(total) if total > 0 else 0
#     acc_personalized = 100 * correct_personalized / float(total) if total > 0 else 0
    
#     # Calculate class-wise accuracy
#     class_acc_global = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct_global.items()}
#     class_acc_personalized = {cls: 100 * correct / class_total[cls] for cls, correct in class_correct_personalized.items()}
    
#     # Calculate balanced accuracy (average of class-wise accuracies)
#     balanced_acc_global = sum(class_acc_global.values()) / len(class_acc_global) if class_acc_global else 0
#     balanced_acc_personalized = sum(class_acc_personalized.values()) / len(class_acc_personalized) if class_acc_personalized else 0

#     logger.info(f'Global Model Accuracy: {acc_global:.2f}% | Personalized Model Accuracy: {acc_personalized:.2f}%')
#     logger.info(f'Global Model Balanced Accuracy: {balanced_acc_global:.2f}% | Personalized Model Balanced Accuracy: {balanced_acc_personalized:.2f}%')

#     # Clean up
#     eval_model.to('cpu')
#     torch.cuda.empty_cache()
    
#     return {
#         "acc_global": acc_global, 
#         "acc_personalized": acc_personalized,
#         "balanced_acc_global": balanced_acc_global,
#         "balanced_acc_personalized": balanced_acc_personalized,
#         "class_acc_global": class_acc_global,
#         "class_acc_personalized": class_acc_personalized
#     }


# def track_trust_scores(trust_scores, round_num, log_dir):
#     """Track client trust scores over rounds.
    
#     Args:
#         trust_scores (dict): Dictionary mapping client IDs to trust scores
#         round_num (int): Current round number
#         log_dir (str): Directory to save trust score logs
        
#     Returns:
#         dict: Statistics about trust scores
#     """
#     import os
#     import json
#     import matplotlib.pyplot as plt
    
#     # Create directory if it doesn't exist
#     os.makedirs(log_dir, exist_ok=True)
    
#     # Save trust scores for current round
#     with open(os.path.join(log_dir, f'trust_scores_round_{round_num}.json'), 'w') as f:
#         json.dump(trust_scores, f)
    
#     # Update trust score history
#     history_file = os.path.join(log_dir, 'trust_score_history.csv')
    
#     # Create file with header if it doesn't exist
#     if not os.path.exists(history_file):
#         with open(history_file, 'w') as f:
#             f.write('round,client_id,trust_score\n')
    
#     # Append current trust scores
#     with open(history_file, 'a') as f:
#         for client_id, score in trust_scores.items():
#             f.write(f'{round_num},{client_id},{score}\n')
    
#     # Calculate statistics
#     scores = list(trust_scores.values())
#     stats = {
#         'mean': np.mean(scores) if scores else 0,
#         'median': np.median(scores) if scores else 0,
#         'min': min(scores) if scores else 0,
#         'max': max(scores) if scores else 0,
#         'std': np.std(scores) if scores else 0,
#         'trusted_clients': sum(1 for s in scores if s >= 0.5),
#         'total_clients': len(scores)
#     }
    
#     # Generate visualization if there are enough rounds
#     if round_num % 10 == 0 and round_num > 0:
#         try:
#             # Load history data
#             import pandas as pd
#             history = pd.read_csv(history_file)
            
#             # Plot trust score evolution
#             plt.figure(figsize=(10, 6))
#             for client in sorted(history['client_id'].unique()):
#                 client_data = history[history['client_id'] == client]
#                 plt.plot(client_data['round'], client_data['trust_score'], 
#                          marker='o', markersize=3, label=f'Client {client}')
            
#             plt.axhline(y=0.5, color='r', linestyle='--', label='Trust Threshold')
#             plt.xlabel('Round')
#             plt.ylabel('Trust Score')
#             plt.title('Client Trust Score Evolution')
#             plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#             plt.tight_layout()
#             plt.savefig(os.path.join(log_dir, f'trust_scores_round_{round_num}.png'))
#             plt.close()
#         except Exception as e:
#             logger.warning(f"Failed to generate trust score visualization: {e}")
    
#     return stats
import copy
import torch
import numpy as np
from torchmetrics import Metric
import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

__all__ = ['evaluate', 'track_trust_scores']

class AccumTensor(Metric):
    def __init__(self, default_value: torch.Tensor):
        super().__init__()
        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor):
        self.val += input_tensor

    def compute(self):
        return self.val


def evaluate(args, model, testloader, device) -> dict:
    """
    Evaluate both global and personalized models on the test set.
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
    class_correct_global = {}
    class_correct_personalized = {}
    class_total = {}

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
            _, predicted_global = torch.max(global_logits, 1)
            _, predicted_personalized = torch.max(personalized_logits, 1)
            
            # Update counters
            total += labels.size(0)
            correct_global += (predicted_global == labels).sum().item()
            correct_personalized += (predicted_personalized == labels).sum().item()
            
            # Update class-wise accuracy
            for label in torch.unique(labels):
                label_idx = (labels == label)
                label_val = label.item()
                
                if label_val not in class_total:
                    class_total[label_val] = 0
                    class_correct_global[label_val] = 0
                    class_correct_personalized[label_val] = 0
                
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
        "class_acc_personalized": class_acc_personalized
    }


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

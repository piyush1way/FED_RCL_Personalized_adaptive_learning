from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from evalers.build import EVALER_REGISTRY
from utils import DatasetSplit
from torch.utils.data import DataLoader
from collections import defaultdict

logger = logging.getLogger(__name__)

@EVALER_REGISTRY.register()
class Evaler:
    def __init__(
        self,
        test_loader: DataLoader,
        device: torch.device,
        args: Dict,
        gallery_loader: DataLoader = None,
        query_loader: DataLoader = None,
        distance_metric: str = 'cosine',
        **kwargs
    ):
        self.args = args
        self.device = device
        self.test_loader = test_loader
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @torch.no_grad()
    def eval(self, model: nn.Module, epoch: int, device: torch.device = None, **kwargs) -> Dict:
        model.eval()
        model_device = next(model.parameters()).device
        device = device or self.device
        model.to(device)

        loss, correct_global, correct_personalized, total = 0, 0, 0, 0

        if isinstance(self.test_loader.dataset, DatasetSplit):
            num_classes = len(self.test_loader.dataset.dataset.classes)
        else:
            num_classes = len(self.test_loader.dataset.classes)

        class_loss = defaultdict(float)
        class_correct_global = defaultdict(int)
        class_correct_personalized = defaultdict(int)
        class_total = defaultdict(int)

        confidence_global = []
        confidence_personalized = []
        
        all_labels = []
        all_global_preds = []
        all_personalized_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)

                has_personalization = hasattr(model, 'personalized_head') or hasattr(model, 'use_personalized_head')
                
                if has_personalization and hasattr(model, 'enable_personalized_mode'):
                    current_mode = getattr(model, 'use_personalized_head', True)
                    
                    model.use_personalized_head = False
                    global_output = model(images)
                    if isinstance(global_output, dict):
                        global_logits = global_output.get("global_logit", global_output.get("logit", None))
                    else:
                        global_logits = global_output
                        
                    model.use_personalized_head = True
                    personalized_output = model(images)
                    if isinstance(personalized_output, dict):
                        personalized_logits = personalized_output.get("personalized_logit", personalized_output.get("logit", None))
                    else:
                        personalized_logits = personalized_output
                        
                    model.use_personalized_head = current_mode
                else:
                    outputs = model(images)
                    
                    if isinstance(outputs, dict):
                        if "global_logit" in outputs and "personalized_logit" in outputs:
                            global_logits = outputs["global_logit"]
                            personalized_logits = outputs["personalized_logit"]
                        elif "logit" in outputs:
                            global_logits = outputs["logit"]
                            personalized_logits = outputs["logit"]
                        else:
                            global_logits = list(outputs.values())[0]
                            personalized_logits = global_logits
                    else:
                        global_logits = outputs
                        personalized_logits = outputs

                global_probs = F.softmax(global_logits, dim=1)
                personalized_probs = F.softmax(personalized_logits, dim=1)
                
                global_conf, predicted_global = torch.max(global_probs, 1)
                personalized_conf, predicted_personalized = torch.max(personalized_probs, 1)
                
                confidence_global.extend(global_conf.cpu().tolist())
                confidence_personalized.extend(personalized_conf.cpu().tolist())
                
                all_labels.extend(labels.cpu().tolist())
                all_global_preds.extend(predicted_global.cpu().tolist())
                all_personalized_preds.extend(predicted_personalized.cpu().tolist())

                total += labels.size(0)
                correct_global += (predicted_global == labels).sum().item()
                correct_personalized += (predicted_personalized == labels).sum().item()

                for label in torch.unique(labels):
                    label_idx = (labels == label)
                    label_val = label.item()
                    
                    class_total[label_val] += label_idx.sum().item()
                    class_correct_global[label_val] += ((predicted_global == labels) & label_idx).sum().item()
                    class_correct_personalized[label_val] += ((predicted_personalized == labels) & label_idx).sum().item()

                this_loss = self.criterion(global_logits, labels)
                loss += this_loss.sum().cpu()

                for label in torch.unique(labels):
                    label_val = label.item()
                    label_mask = (labels == label_val)
                    class_loss[label_val] += this_loss[label_mask].sum().cpu().item()

        acc_global = 100.0 * correct_global / total if total > 0 else 0
        acc_personalized = 100.0 * correct_personalized / total if total > 0 else 0
        
        class_acc_global = {cls: 100.0 * correct / class_total[cls] for cls, correct in class_correct_global.items()}
        class_acc_personalized = {cls: 100.0 * correct / class_total[cls] for cls, correct in class_correct_personalized.items()}
        
        avg_class_loss = {cls: loss / class_total[cls] for cls, loss in class_loss.items()}
        total_loss = loss / total if total > 0 else 0
        
        balanced_acc_global = sum(class_acc_global.values()) / len(class_acc_global) if class_acc_global else 0
        balanced_acc_personalized = sum(class_acc_personalized.values()) / len(class_acc_personalized) if class_acc_personalized else 0
        
        avg_confidence_global = sum(confidence_global) / len(confidence_global) if confidence_global else 0
        avg_confidence_personalized = sum(confidence_personalized) / len(confidence_personalized) if confidence_personalized else 0

        results = {
            "acc": acc_global,
            "acc_personalized": acc_personalized,
            "balanced_acc_global": balanced_acc_global,
            "balanced_acc_personalized": balanced_acc_personalized,
            "class_acc_global": class_acc_global,
            "class_acc_personalized": class_acc_personalized,
            "loss": total_loss,
            "class_loss": avg_class_loss,
            "avg_confidence_global": avg_confidence_global,
            "avg_confidence_personalized": avg_confidence_personalized,
            "predictions": {
                "labels": all_labels,
                "global_preds": all_global_preds,
                "personalized_preds": all_personalized_preds
            }
        }

        logger.info(f"[Epoch {epoch}] Global Acc: {acc_global:.2f}%, Personalized Acc: {acc_personalized:.2f}%")
        logger.info(f"[Epoch {epoch}] Balanced Global Acc: {balanced_acc_global:.2f}%, Balanced Personalized Acc: {balanced_acc_personalized:.2f}%")
        
        return results

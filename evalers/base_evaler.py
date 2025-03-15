# from pathlib import Path
# from typing import Callable, Dict, Tuple, Union, List, Type
# from argparse import Namespace
# from collections import defaultdict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.multiprocessing as mp
# import tqdm
# import wandb
# import gc

# import pickle, os
# import numpy as np

# import logging
# logger = logging.getLogger(__name__)


# import time, io, copy

# from evalers.build import EVALER_REGISTRY

# from servers import Server
# from clients import Client

# from utils import DatasetSplit, get_dataset
# from utils.logging_utils import AverageMeter

# from torch.utils.data import DataLoader

# from utils import terminate_processes, initalize_random_seed
# from omegaconf import DictConfig

# import umap.umap_ as umap
# from sklearn import metrics
# import matplotlib.pyplot as plt

# import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler



# @EVALER_REGISTRY.register()
# class Evaler():

#     def __init__(self,
#                  test_loader: torch.utils.data.DataLoader,
#                 device: torch.device,
#                 args: DictConfig,
#                 gallery_loader: torch.utils.data.DataLoader = None,
#                 query_loader: torch.utils.data.DataLoader = None,
#                 distance_metric: str = 'cosine',
#                 **kwargs) -> None:

#         self.args = args
#         self.device = device

#         self.test_loader = test_loader
#         self.gallery_loader = gallery_loader
#         self.query_loader = query_loader
#         self.criterion = nn.CrossEntropyLoss(reduction = 'none')


#     @torch.no_grad()
#     def eval(self, model: nn.Module, epoch: int, device: torch.device = None, **kwargs) -> Dict:

#         model.eval()
#         model_device = next(model.parameters()).device
#         if device is None:
#             device = self.device
#         model.to(device)
#         loss, correct, total = 0, 0, 0

#         if type(self.test_loader.dataset) == DatasetSplit:
#             C = len(self.test_loader.dataset.dataset.classes)
#         else:
#             C = len(self.test_loader.dataset.classes)

#         class_loss, class_correct, class_total = torch.zeros(C), torch.zeros(C), torch.zeros(C)

#         logits_all, labels_all = [], []


#         with torch.no_grad():
#             # for images, labels in self.loaders["test"]:
#             for idx, (images, labels) in enumerate(self.test_loader):
#                 images, labels = images.to(device), labels.to(device)

#                 results = model(images)
#                 _, predicted = torch.max(results["logit"].data, 1) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#                 bin_labels = labels.bincount()
#                 class_total[:bin_labels.size(0)] += bin_labels.cpu()
#                 bin_corrects = labels[(predicted == labels)].bincount()
#                 class_correct[:bin_corrects.size(0)] += bin_corrects.cpu()

#                 this_loss = self.criterion(results["logit"], labels)
#                 loss += this_loss.sum().cpu()

#                 for class_idx, bin_label in enumerate(bin_labels):
#                     class_loss[class_idx] += this_loss[(labels.cpu() == class_idx)].sum().cpu()

#                 logits_all.append(results["logit"].data.cpu())
#                 labels_all.append(labels.cpu())

#         logits_all = torch.cat(logits_all)
#         labels_all = torch.cat(labels_all)

#         scores = F.softmax(logits_all, 1)

#         acc = 100. * correct / float(total)
#         class_acc = 100. * class_correct / class_total
        
#         loss = loss / float(total)
#         class_loss = class_loss / class_total

#         model.train()
#         results = {
#             "acc": acc,
#             'class_acc': class_acc,
#             'loss': loss,
#             'class_loss' : class_loss,
#         }
        
#         return results

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
        """
        Evaluate model performance on the test set.

        Args:
            model (nn.Module): The trained model.
            epoch (int): The current epoch number.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Dict: A dictionary containing accuracy, class-wise accuracy, and loss.
        """
        model.eval()
        model_device = next(model.parameters()).device
        device = device or self.device
        model.to(device)

        loss, correct_global, correct_personalized, total = 0, 0, 0, 0

        if isinstance(self.test_loader.dataset, DatasetSplit):
            num_classes = len(self.test_loader.dataset.dataset.classes)
        else:
            num_classes = len(self.test_loader.dataset.classes)

        class_loss, class_correct_global, class_correct_personalized, class_total = (
            torch.zeros(num_classes),
            torch.zeros(num_classes),
            torch.zeros(num_classes),
            torch.zeros(num_classes),
        )

        logits_all, labels_all = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)

                # Evaluate global model outputs
                global_results = model(images)
                if not isinstance(global_results, dict):
                    raise ValueError("Model output must be a dictionary containing 'feature' and 'logit' keys.")

                # Ensure global_results contains 'feature' and 'logit' keys
                if "feature" not in global_results or "logit" not in global_results:
                    raise KeyError("Model output must contain 'feature' and 'logit' keys.")

                # Evaluate personalized model outputs
                if hasattr(model, 'forward_classifier'):
                    personalized_results = model.forward_classifier(global_results["feature"])
                else:
                    # Fallback to using the global model's logits if personalized classifier is not available
                    personalized_results = global_results["logit"]

                # Compute predictions
                _, predicted_global = torch.max(global_results["logit"].data, 1)
                _, predicted_personalized = torch.max(personalized_results.data, 1)

                # Update metrics
                total += labels.size(0)
                correct_global += (predicted_global == labels).sum().item()
                correct_personalized += (predicted_personalized == labels).sum().item()

                # Update class-wise metrics
                bin_labels = labels.bincount()
                class_total[: bin_labels.size(0)] += bin_labels.cpu()

                bin_corrects_global = labels[(predicted_global == labels)].bincount()
                bin_corrects_personalized = labels[(predicted_personalized == labels)].bincount()

                class_correct_global[: bin_corrects_global.size(0)] += bin_corrects_global.cpu()
                class_correct_personalized[: bin_corrects_personalized.size(0)] += bin_corrects_personalized.cpu()

                # Compute loss
                this_loss = self.criterion(global_results["logit"], labels)
                loss += this_loss.sum().cpu()

                # Update class-wise loss
                for class_idx, bin_label in enumerate(bin_labels):
                    class_loss[class_idx] += this_loss[(labels.cpu() == class_idx)].sum().cpu()

                # Store logits and labels for additional metrics
                logits_all.append(global_results["logit"].data.cpu())
                labels_all.append(labels.cpu())

        # Concatenate logits and labels
        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)
        scores = F.softmax(logits_all, 1)

        # Compute metrics
        acc_global = 100.0 * correct_global / float(total)
        acc_personalized = 100.0 * correct_personalized / float(total)
        class_acc_global = 100.0 * class_correct_global / class_total
        class_acc_personalized = 100.0 * class_correct_personalized / class_total
        loss = loss / float(total)
        class_loss = class_loss / class_total

        # Prepare results dictionary
        results = {
            "acc_global": acc_global,
            "acc_personalized": acc_personalized,
            "class_acc_global": class_acc_global,
            "class_acc_personalized": class_acc_personalized,
            "loss": loss,
            "class_loss": class_loss,
        }

        # Log results
        logger.info(f"[Epoch {epoch}] Global Test Accuracy: {acc_global:.2f}% | Personalized Test Accuracy: {acc_personalized:.2f}%")
        return results

#!/usr/bin/env python
# coding: utf-8
import copy
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import logging
from utils import *
from utils.loss import KL_u_p_loss
from utils.metrics import evaluate
from models import build_encoder
from utils.logging_utils import AverageMeter
from clients.build import CLIENT_REGISTRY
from clients import Client

logger = logging.getLogger(__name__)

@CLIENT_REGISTRY.register()
class RCLClient(Client):
    def __init__(self, args, client_index, model):
        self.args = args
        self.client_index = client_index
        self.loader = None

        # Initialize trust filtering settings
        trust_filtering_config = getattr(args.client, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)

        # Initialize personalization settings
        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", True)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", True)

        self.model = model
        self.global_model = copy.deepcopy(model)

        self.rcl_criterions = {'scl': None, 'penalty': None, }
        args_rcl = args.client.rcl_loss
        self.global_epoch = 0

        self.pairs = {}
        for pair in args_rcl.pairs:
            self.pairs[pair.name] = pair
            self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
        
        self.criterion = nn.CrossEntropyLoss()

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        """Initialize client model, dataset, and optimizer"""
        self._update_model(state_dict)
        self._update_global_model(state_dict)

        # Freeze the global model
        for param in self.global_model.parameters():
            param.requires_grad = False

        # Apply personalization settings
        if self.enable_personalization:
            if self.freeze_backbone:
                self.model.freeze_backbone()
            if self.adaptive_layer_freezing:
                self.model.setup_adaptive_freezing()

        self.device = device
        self.trainer = trainer
        self.global_epoch = global_epoch

        # Create DataLoader
        train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
            if self.args.dataset.num_instances > 0 else None

        self.loader = DataLoader(
            local_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )

        # Optimizer & Scheduler
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=local_lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.wd
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
        )

    def compute_trust_score(self, local_state_dict):
        """
        Compute trust score for this client based on various metrics.
        Higher score means the client's updates are more reliable.
        """
        trust_score = 1.0  # Default trust score
        
        # Add your trust score computation logic here
        # For example:
        # 1. Check data quality
        # 2. Check update magnitude
        # 3. Check historical performance
        # 4. Check for adversarial behavior
        
        return max(0.0, min(1.0, trust_score))  # Ensure score is between 0 and 1

    def adapt_learning_rate(self, gradients):
        """Adjust learning rate based on gradient variance"""
        variance = torch.var(torch.stack([torch.norm(g) for g in gradients]))
        new_lr = max(self.min_lr, min(self.max_lr, self.base_lr / (1 + variance)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def local_train(self, global_epoch, **kwargs):
        """Performs local training on the client's dataset"""

        self.global_epoch = global_epoch
        self.model.to(self.device)
        self.global_model.to(self.device)

        scaler = GradScaler()
        start_time = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')

        gradients = []

        for local_epoch in range(self.args.trainer.local_epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                gradients.append([p.grad for p in self.model.parameters() if p.grad is not None])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()

                loss_meter.update(loss.item(), images.size(0))

            self.scheduler.step()

            # Adaptive Learning Rate Adjustment
            if self.adaptive_lr:
                new_lr = self.adapt_learning_rate(gradients)
                logger.info(f"[C{self.client_index}] Adaptive LR Adjusted: {new_lr:.5f}")

        end_time = time.time()

        # Compute trust score for client updates
        trust_score = self.compute_trust_score(self.model.state_dict())

        logger.info(f"[C{self.client_index}] Training Complete. Time: {end_time - start_time:.2f}s, Loss: {loss_meter.avg:.3f}, Trust Score: {trust_score:.3f}")

        # Filter out low-trust clients
        if self.enable_trust_filtering and trust_score < self.trust_threshold:
            logger.warning(f"[C{self.client_index}] Skipped in Aggregation (Trust Score {trust_score:.3f} < {self.trust_threshold})")
            return None, None

        self.model.to('cpu')
        self.global_model.to('cpu')

        gc.collect()

        return self.model.state_dict(), {"loss": float(loss_meter.avg), "trust_score": trust_score}

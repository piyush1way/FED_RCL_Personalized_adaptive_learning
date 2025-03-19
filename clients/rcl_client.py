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
from utils.loss import KL_u_p_loss, RelaxedContrastiveLoss
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

        trust_filtering_config = getattr(args.client, "trust_filtering", {})
        self.enable_trust_filtering = getattr(trust_filtering_config, "enable", False)
        self.trust_threshold = getattr(trust_filtering_config, "trust_threshold", 0.5)
        
        self.rounds_trained = 0
        self.previous_model_state = None
        self.update_history = []

        personalization_config = getattr(args.client, "personalization", {})
        self.enable_personalization = getattr(personalization_config, "enable", False)
        self.freeze_backbone = getattr(personalization_config, "freeze_backbone", False)
        self.adaptive_layer_freezing = getattr(personalization_config, "adaptive_layer_freezing", False)

        self.enable_cyclical_lr = True
        self.base_lr = 0.001
        self.max_lr = 0.1
        self.step_size = 10
        
        self.enable_fedprox = True
        self.fedprox_mu = 0.005

        self.enable_distillation = True
        self.distillation_temp = 3.0
        self.distillation_weight = 0.7

        self.model = model
        self.global_model = copy.deepcopy(model)
        self.device = torch.device("cpu")

        self.rcl_criterions = {'scl': None, 'penalty': None}
        args_rcl = getattr(args.client, "rcl_loss", None)
        if args_rcl:
            self.pairs = {}
            for pair in args_rcl.pairs:
                self.pairs[pair.name] = pair
                self.rcl_criterions[pair.name] = CLLoss(pair=pair, **args_rcl)
        else:
            self.pairs = {}
            
        self.global_epoch = 0
        self.criterion = nn.CrossEntropyLoss()
        
        self.relaxed_contrastive_loss = RelaxedContrastiveLoss(
            temperature=0.2,
            lambda_penalty=0.03,
            similarity_threshold=0.4
        )
        
        self.ce_loss_avg = 0.0
        self.rcl_loss_avg = 0.0
        self.distillation_loss_avg = 0.0
        self.fedprox_loss_avg = 0.0
        
        self.ewc_importance = 5000
        self.ewc_enabled = True
        self.fisher_information = None
        self.optimal_parameters = None

    def setup(self, state_dict, device, local_dataset, global_epoch, local_lr, trainer, **kwargs):
        self.device = device
        self.rounds_trained += 1
        
        if self.enable_trust_filtering and hasattr(self.model, 'state_dict'):
            self.previous_model_state = {k: v.to(device) for k, v in self.model.state_dict().items()}
        
        if self.ewc_enabled and self.fisher_information is not None:
            self.optimal_parameters = {k: v.clone().detach() for k, v in self.model.state_dict().items() 
                                      if 'personalized_head' not in k}
        
        self._update_model(state_dict)
        self._update_global_model(state_dict)

        self.model.to(self.device)
        self.global_model.to(self.device)

        for param in self.global_model.parameters():
            param.requires_grad = False

        if self.enable_personalization:
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            if hasattr(model_module, 'enable_personalized_mode'):
                model_module.enable_personalized_mode()
            elif hasattr(model_module, 'use_personalized_head'):
                model_module.use_personalized_head = True
                
            if self.freeze_backbone and hasattr(model_module, 'freeze_backbone'):
                if self.rounds_trained < 3:
                    model_module.unfreeze_backbone()
                else:
                    model_module.freeze_backbone()
                
            if self.adaptive_layer_freezing and hasattr(model_module, 'setup_adaptive_freezing'):
                freeze_ratio = max(0.0, 0.8 - (self.rounds_trained / 50.0))
                model_module.setup_adaptive_freezing(freeze_ratio=freeze_ratio)

        self.trainer = trainer
        self.global_epoch = global_epoch

        if local_dataset is None or len(local_dataset) == 0:
            logger.warning(f"[C{self.client_index}] Empty dataset provided. Creating a dummy loader.")
            self.loader = None
        else:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances) \
                if hasattr(self.args.dataset, 'num_instances') and self.args.dataset.num_instances > 0 else None
                
            self.loader = DataLoader(
                local_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                shuffle=train_sampler is None,
                num_workers=0,
                pin_memory=False,
                drop_last=True
            )

        if self.enable_cyclical_lr:
            cycle_step = self.rounds_trained % (2 * self.step_size)
            x = abs(cycle_step / self.step_size - 1)
            adjusted_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
            logger.info(f"[C{self.client_index}] Cyclical LR: {adjusted_lr:.6f} (cycle step: {cycle_step})")
            local_lr = adjusted_lr
        else:
            logger.info(f"[C{self.client_index}] Using base LR: {local_lr:.6f}")

        if self.enable_personalization:
            if hasattr(model_module, 'get_local_params'):
                personalized_params = []
                backbone_params = []
                for name, param in self.model.named_parameters():
                    if 'personalized_head' in name or 'projection_head' in name:
                        personalized_params.append(param)
                    else:
                        backbone_params.append(param)
                
                if personalized_params:
                    param_groups = [
                        {'params': personalized_params, 'lr': local_lr},
                        {'params': backbone_params, 'lr': local_lr * 0.1}
                    ]
                    self.optimizer = torch.optim.SGD(
                        param_groups,
                        momentum=self.args.optimizer.momentum,
                        weight_decay=self.args.optimizer.wd
                    )
                else:
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=local_lr,
                        momentum=self.args.optimizer.momentum,
                        weight_decay=self.args.optimizer.wd
                    )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=local_lr,
                    momentum=self.args.optimizer.momentum,
                    weight_decay=self.args.optimizer.wd
                )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=local_lr,
                momentum=self.args.optimizer.momentum,
                weight_decay=self.args.optimizer.wd
            )
            
        if self.rounds_trained < 5:
            self.local_epochs = min(30, self.args.trainer.local_epochs * 3)
        else:
            self.local_epochs = self.args.trainer.local_epochs
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch
        )

    def compute_fedprox_term(self):
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)**2
        return self.fedprox_mu * proximal_term / 2

    def compute_distillation_loss(self, student_logits, teacher_logits, features=None, teacher_features=None):
        temp = self.distillation_temp
        soft_targets = F.softmax(teacher_logits / temp, dim=1)
        log_probs = F.log_softmax(student_logits / temp, dim=1)
        logit_distillation = -(soft_targets * log_probs).sum(dim=1).mean() * (temp ** 2)
        
        feature_distillation = 0.0
        if features is not None and teacher_features is not None:
            features_norm = F.normalize(features, dim=1)
            teacher_features_norm = F.normalize(teacher_features, dim=1)
            feature_distillation = (1 - (features_norm * teacher_features_norm).sum(dim=1).mean())
            
        return logit_distillation + 0.5 * feature_distillation

    def compute_ewc_loss(self):
        if not self.ewc_enabled or self.fisher_information is None or self.optimal_parameters is None:
            return 0.0
            
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_parameters:
                if 'personalized_head' not in name:
                    loss += (self.fisher_information[name] * (param - self.optimal_parameters[name]).pow(2)).sum()
                    
        return self.ewc_importance * loss

    def compute_trust_score(self):
        if self.previous_model_state is None:
            return 1.0
        
        current_state = self.model.state_dict()
        
        update_norm = 0.0
        param_count = 0
        
        for key in current_state:
            if 'personalized_head' not in key and key in self.previous_model_state:
                current_param = current_state[key].to(self.device).float()
                prev_param = self.previous_model_state[key].to(self.device).float()
                diff = current_param - prev_param
                update_norm += torch.norm(diff).item() ** 2
                param_count += diff.numel()
        
        if param_count > 0:
            update_norm = (update_norm / param_count) ** 0.5
        
        self.update_history.append(update_norm)
        if len(self.update_history) > 5:
            self.update_history.pop(0)
            
        update_variance = np.var(self.update_history) if len(self.update_history) > 1 else 0.0
        
        magnitude_score = 1.0 / (1.0 + update_norm)
        consistency_score = 1.0 / (1.0 + update_variance)
        
        trust_score = 0.7 * magnitude_score + 0.3 * consistency_score
        
        self.previous_model_state = {k: v.detach().clone().to(self.device) for k, v in current_state.items()}
        
        return max(0.0, min(1.0, trust_score))

    def compute_fisher_information(self):
        if not self.ewc_enabled or self.loader is None:
            return
            
        fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'personalized_head' not in name:
                fisher_information[name] = torch.zeros_like(param)
                
        self.model.eval()
        for images, labels in self.loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.model.zero_grad()
            output = self.model(images)
            if isinstance(output, dict):
                logits = output["logit"]
            else:
                logits = output
                
            log_probs = F.log_softmax(logits, dim=1)
            samples = torch.multinomial(torch.exp(log_probs), 1).squeeze()
            loss = F.nll_loss(log_probs, samples)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and 'personalized_head' not in name:
                    fisher_information[name] += param.grad.pow(2).detach()
                    
        for name in fisher_information:
            fisher_information[name] /= len(self.loader) if len(self.loader) > 0 else 1.0
            
        self.fisher_information = fisher_information
        self.optimal_parameters = {name: param.clone().detach() for name, param in self.model.named_parameters() 
                                  if name in fisher_information}
        self.model.train()

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        scaler = GradScaler(enabled=self.device.type == "cuda")
        start_time = time.time()
        loss_meter = AverageMeter('Loss', ':.4f')
        ce_loss_meter = AverageMeter('CE Loss', ':.4f')
        rcl_loss_meter = AverageMeter('RCL Loss', ':.4f')
        distillation_loss_meter = AverageMeter('Distill Loss', ':.4f')
        fedprox_loss_meter = AverageMeter('FedProx Loss', ':.4f')
        ewc_loss_meter = AverageMeter('EWC Loss', ':.4f')

        if self.loader is None or len(self.loader) == 0:
            logger.warning(f"[C{self.client_index}] No data for training!")
            return None, None

        for local_epoch in range(self.local_epochs):
            epoch_loss = 0.0
            samples_processed = 0
            
            for images, labels in self.loader:
                if len(images) < 2:
                    continue
                    
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.model.zero_grad()

                with autocast(enabled=self.device.type == "cuda"):
                    output = self.model(images, get_projection=True)
                    
                    if isinstance(output, dict):
                        logits = output["logit"] if "logit" in output else output.get("personalized_logit", output)
                        features = output.get("feature", None)
                        projection = output.get("projection", None)
                    else:
                        logits = output
                        features = None
                        projection = None
                    
                    ce_loss = self.criterion(logits, labels)
                    
                    rcl_loss = 0.0
                    if hasattr(self.args.client, 'rcl_loss') and getattr(self.args.client.rcl_loss, 'weight', 0) > 0:
                        contrastive_features = projection if projection is not None else features
                        
                        if contrastive_features is not None and len(contrastive_features) >= 2:
                            rcl_weight = getattr(self.args.client.rcl_loss, 'weight', 0.1)
                            rcl_loss = self.relaxed_contrastive_loss(contrastive_features, labels) * rcl_weight
                    
                    distillation_loss = 0.0
                    if self.enable_distillation:
                        with torch.no_grad():
                            global_output = self.global_model(images)
                            if isinstance(global_output, dict):
                                global_logits = global_output.get("global_logit", global_output.get("logit", None))
                                global_features = global_output.get("feature", None)
                            else:
                                global_logits = global_output
                                global_features = None
                                
                        if global_logits is not None:
                            distillation_loss = self.compute_distillation_loss(
                                logits, global_logits, features, global_features
                            ) * self.distillation_weight
                    
                    fedprox_loss = 0.0
                    if self.enable_fedprox:
                        fedprox_loss = self.compute_fedprox_term()
                        
                    ewc_loss = 0.0
                    if self.ewc_enabled and self.fisher_information is not None:
                        ewc_loss = self.compute_ewc_loss()
                    
                    loss = ce_loss + rcl_loss + distillation_loss + fedprox_loss + ewc_loss
                    
                    ce_loss_meter.update(ce_loss.item(), images.size(0))
                    if rcl_loss > 0:
                        rcl_loss_meter.update(rcl_loss.item(), images.size(0))
                    if distillation_loss > 0:
                        distillation_loss_meter.update(distillation_loss.item(), images.size(0))
                    if fedprox_loss > 0:
                        fedprox_loss_meter.update(fedprox_loss.item(), images.size(0))
                    if ewc_loss > 0:
                        ewc_loss_meter.update(ewc_loss.item(), images.size(0))
                
                if not torch.isfinite(loss):
                    logger.warning(f"[C{self.client_index}] Loss is {loss}, skipping batch")
                    continue
                    
                if self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                if self.device.type == "cuda":
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                batch_size = images.size(0)
                loss_meter.update(loss.item(), batch_size)
                epoch_loss += loss.item() * batch_size
                samples_processed += batch_size
            if samples_processed > 0:
                logger.info(f"[C{self.client_index}] Epoch {local_epoch+1}/{self.local_epochs}, Loss: {epoch_loss/samples_processed:.4f}")
            else:
                logger.warning(f"[C{self.client_index}] No samples processed in epoch {local_epoch+1}")
            
            self.scheduler.step()
        end_time = time.time()
        training_time = end_time - start_time
        
        self.ce_loss_avg = ce_loss_meter.avg
        self.rcl_loss_avg = rcl_loss_meter.avg
        self.distillation_loss_avg = distillation_loss_meter.avg
        self.fedprox_loss_avg = fedprox_loss_meter.avg
        self.ewc_loss_avg = ewc_loss_meter.avg if hasattr(self, 'ewc_loss_avg') else 0.0
        
        if self.ewc_enabled and self.rounds_trained > 1:
            self.compute_fisher_information()
        
        trust_score = self.compute_trust_score() if self.enable_trust_filtering else 1.0
        logger.info(f"[C{self.client_index}] Training Complete. Time: {training_time:.2f}s, Loss: {loss_meter.avg:.4f}, Trust Score: {trust_score:.3f}")
        if self.enable_personalization:
            if isinstance(self.model, torch.nn.DataParallel):
                model_module = self.model.module
            else:
                model_module = self.model
                
            if hasattr(model_module, 'get_global_params'):
                global_params = model_module.get_global_params()
                return_dict = {k: v.cpu() for k, v in global_params.items()}
            else:
                return_dict = {k: v.cpu() for k, v in self.model.state_dict().items() 
                              if 'personalized_head' not in k}
        else:
            return_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        if self.enable_trust_filtering and trust_score < self.trust_threshold:
            logger.warning(f"[C{self.client_index}] Skipped in Aggregation (Trust Score {trust_score:.3f} < {self.trust_threshold})")
            return None, None
        self.model = self.model.cpu()
        self.global_model = self.global_model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return return_dict, {"loss": float(loss_meter.avg), 
                            "trust_score": trust_score, 
                            "ce_loss": float(self.ce_loss_avg),
                            "rcl_loss": float(self.rcl_loss_avg),
                            "distillation_loss": float(self.distillation_loss_avg),
                            "fedprox_loss": float(self.fedprox_loss_avg),
                            "ewc_loss": float(self.ewc_loss_avg) if hasattr(self, 'ewc_loss_avg') else 0.0}

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from utils.helper import *
from utils.logging_utils import AverageMeter
import math

__all__ = [
    'MultiLabelCrossEntropyLoss', 
    'CLLoss', 
    'KL_u_p_loss', 
    'compute_personalized_loss', 
    'RelaxedContrastiveLoss',
    'MultiLevelContrastiveLoss',
    'HybridPersonalizationLoss',
    'TrustBasedLoss',
    'KnowledgeDistillationLoss',
    'FedProxLoss',
    'EWCLoss'
]

class MultiLabelCrossEntropyLoss(nn.Module):
    def __init__(self, eps: float=0, alpha: float=0.2, topk_pos: int=-1, temp: float=1., **kwargs):
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.topk_pos = topk_pos
        self.temp = temp

    def __repr__(self):
        return "MultiLabelCrossEntropyLoss(eps={}, alpha={})".format(self.eps, self.alpha)

    def forward(self, input: torch.Tensor, targets: torch.Tensor, reduction: bool = True, beta: float = None) -> torch.Tensor:
        N, C = input.size()
        E = self.eps

        input[input==np.inf] = -np.inf

        if beta is not None:
            weights = torch.ones_like(targets)
            weights[targets==0] = beta
            input += torch.log(weights)/self.temp

        log_probs = F.log_softmax(input, dim=1)
        loss_ = (-targets * log_probs)
        loss_[loss_==np.inf] = 0.
        loss_[loss_==-np.inf] = 0.
        loss_[loss_.isnan()] = 0.
        loss = loss_.sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        if reduction:
            loss = loss.sum() / non_zero_cnt
        else:
            loss = loss

        return loss

    
class CLLoss(nn.Module):
    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, pos_loss_type=None, neg_loss_type=None,
                 threshold=1, **kwargs):
        super(CLLoss, self).__init__()
        self.pair = pair
        
        self.topk_pos = self.pair.get('topk_pos') or topk_pos
        self.topk_neg = self.pair.get('topk_neg') or topk_neg
        self.pos_sample_type = self.pair.get('pos_sample_type') or pos_sample_type
        self.neg_sample_type = self.pair.get('neg_sample_type') or neg_sample_type
        
        self.temp = self.pair.get('temp') or temp
        
        self.loss_type = self.pair.get('loss_type') or loss_type
        self.pos_loss_type = self.pair.get('pos_loss_type') or pos_loss_type 
        self.neg_loss_type = self.pair.get('neg_loss_type') or neg_loss_type 
        self.beta = self.pair.get('beta') or beta

        self.threshold = self.pair.get('threshold') or threshold

        self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos, temp=temp, eps=eps)

    def __set_num_classes__(self, num_classes):
        self.num_classes = num_classes

    def __repr__(self):
        return "{}(topk_pos={}, topk_neg={}, temp={}, crit={}), pair={})".format(
            type(self).__name__, self.topk_pos, self.topk_neg, self.temp, self.criterion, self.pair)

    def get_classwise_mask(self, target):
        B = target.size(0)
        classwise_mask = target.expand(B, B).eq(target.expand(B, B).T)
        return classwise_mask
    
    def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, topk_pos=None, labels=None):
        B = sim.size(0)
        if B < 2:
            return torch.tensor([], device=sim.device)

        sim_neg = sim.clone()

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        if 'unsupervised' in neg_loss_type:
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        if self.neg_sample_type:
            if self.neg_sample_type == 'intra_class_thresholding':
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = np.inf
                sim_neg[torch.eye(B)==1] = np.inf
                
                k = min(topk_pos, B-1) if topk_pos > 0 else 1
                if k > 0:
                    sim_neg = torch.topk(sim_neg, k, dim=1, largest=False)[0]
                    idx = sim_neg < self.threshold
                    sim_neg[idx] = -1
                    sim_neg[sim_neg == np.inf] = -1
                else:
                    return torch.tensor([], device=sim.device)
            else:
                raise ValueError

        else:
            sim_neg[pos_mask==1] = -np.inf
            k = min(topk_neg, B-1) if topk_neg > 0 else 1
            if k > 0:
                sim_neg = torch.topk(sim_neg, k, dim=1, largest=True)[0]
            else:
                return torch.tensor([], device=sim.device)

        return sim_neg
    
    def get_topk_pos(self, sim, topk_pos=None, labels=None, uncertainty=None):
        B = sim.size(0)
        if B < 2:
            return torch.tensor([], device=sim.device)

        sim_pos = sim.clone()

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        sim_pos[pos_mask==0] = np.inf
        
        k = min(topk_pos, (pos_mask.sum(dim=1) - 1).max().item()) if topk_pos > 0 else 1
        if k > 0:
            sim_pos, inds = torch.topk(sim_pos, k, dim=1, largest=False)
        else:
            return torch.tensor([], device=sim.device)

        return sim_pos

    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, name="loss1"):
        if old_feat.size(0) < 2 or new_feat.size(0) < 2:
            return torch.tensor(0.0, device=old_feat.device)

        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)

        B, C = new_feat.size()    

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}
        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None

        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos
        if topk_neg is None:
            topk_neg = self.topk_neg
            
        topk_pos = min(topk_pos, B-1) if topk_pos > 0 else 1
        topk_neg = min(topk_neg, B-1) if topk_neg > 0 else 1

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        for pair_type in all_pair_types:
            try:
                sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
                sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, topk_pos=topk_pos, labels=target)
            except Exception as e:
                return torch.tensor(0.0, device=old_feat.device)

        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = [], []
        
        for pos_name in pair['pos'].split(' '):
            if pos_name in sim_poss and sim_poss[pos_name].numel() > 0:
                pair_poss.append(sim_poss[pos_name])
        
        for neg_name in pair['neg'].split(' '):
            if neg_name in sim_negs and sim_negs[neg_name].numel() > 0:
                pair_negs.append(sim_negs[neg_name])
        
        if not pair_poss or not pair_negs:
            return torch.tensor(0.0, device=old_feat.device)
        
        pair_poss = torch.cat(pair_poss, 1)
        pair_negs = torch.cat(pair_negs, 1)
        
        pair_poss_ = pair_poss.unsqueeze(2)
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1], 1)
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2)
        
        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1
        
        scaled_inputs = pair_all_.reshape(-1, pair_all_.size(2)) / self.temp
        
        try:
            loss = self.criterion(
                input=scaled_inputs,
                targets=binary_zero_labels_.reshape(-1, pair_all_.size(2)),
                reduction=reduction, 
                beta=self.beta
            )
            
            if loss.numel() > 1:
                loss = loss.reshape(B, -1).mean(1)
                
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)) or torch.any(loss > 100):
                return torch.tensor(1.0, device=old_feat.device)
                
            return loss
        except Exception:
            return torch.tensor(1.0, device=old_feat.device)


class RelaxedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, beta=1.0, lambda_threshold=0.7):
        """
        Implementation of the Relaxed Contrastive Loss from the FedRCL paper
        
        Args:
            temperature: Temperature parameter τ (default: 0.05 as in paper)
            beta: Weight for the divergence penalty term (default: 1.0 as in paper)
            lambda_threshold: Similarity threshold λ for identifying too-similar pairs (default: 0.7 as in paper)
        """
        super(RelaxedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.lambda_threshold = lambda_threshold
        
    def forward(self, features, labels):
        # Handle type conversion first to avoid dtype issues
        if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure features are float type for numerical operations
        if features.dtype != torch.float32 and features.dtype != torch.float64:
            features = features.float()
        
        # Ensure labels are long type for indexing
        if labels.dtype != torch.int64 and labels.dtype != torch.long:
            labels = labels.long()
        
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)
            
        # Normalize feature vectors for numerical stability
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix with scaling
        sim_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Apply temperature scaling in a numerically stable way
        # Subtract max for numerical stability before exponentiation
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_stable = sim_matrix - sim_max
        
        # Create mask for positive pairs (same class)
        labels_expand = labels.expand(batch_size, batch_size)
        pos_mask = labels_expand.eq(labels_expand.t()).float()
        
        # Remove self-connections
        eye_mask = torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask - eye_mask
        neg_mask = 1 - pos_mask - eye_mask
        
        # If no positive pairs exist in the batch
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Use the log-sum-exp trick for numerical stability
        exp_sim = torch.exp(sim_matrix_stable)
        
        # For numerical stability, mask out self-similarity
        exp_sim = exp_sim * (1 - eye_mask)
        
        # Compute denominators (sum over all samples except self)
        denominators = exp_sim.sum(dim=1, keepdim=True)
        
        # Use logsumexp for numerical stability
        log_denominators = torch.log(denominators + 1e-12) + sim_max
        
        # Compute supervised contrastive loss using log-space operations
        eps = 1e-12  # Small epsilon to avoid numerical issues
        scl_loss = 0
        valid_pairs = 0
        
        for i in range(batch_size):
            pos_indices = torch.nonzero(pos_mask[i]).squeeze()
            if pos_indices.numel() > 0:  # If there are positive samples
                if pos_indices.numel() == 1:
                    pos_indices = pos_indices.unsqueeze(0)
                
                # Compute loss for each positive pair
                for j in pos_indices:
                    # Compute log(numerator/denominator) directly
                    log_numerator = sim_matrix[i, j]
                    scl_loss -= (log_numerator - log_denominators[i])
                    valid_pairs += 1
        
        # Normalize by number of positive pairs
        if valid_pairs > 0:
            scl_loss = scl_loss / valid_pairs
        
        # Relaxation term: penalty for too-similar positive pairs
        # Find positive pairs with similarity > lambda
        high_sim_pos = (sim_matrix * self.temperature > self.lambda_threshold) & (pos_mask > 0)
        
        relaxation_loss = 0
        valid_high_sim_pairs = 0
        
        if high_sim_pos.sum() > 0:
            for i in range(batch_size):
                high_sim_indices = torch.nonzero(high_sim_pos[i]).squeeze()
                if high_sim_indices.numel() > 0:
                    if high_sim_indices.numel() == 1:
                        high_sim_indices = high_sim_indices.unsqueeze(0)
                    
                    # Use logsumexp for numerical stability
                    high_sim_logits = sim_matrix_stable[i, high_sim_indices]
                    
                    # Add baseline term - ensure it's a tensor
                    baseline_logit = torch.tensor(1.0/self.temperature, device=features.device) - sim_max[i]
                    high_sim_logits_with_baseline = torch.cat([high_sim_logits, baseline_logit.reshape(1)])
                    
                    # Compute log-sum-exp directly
                    max_logit = torch.max(high_sim_logits_with_baseline)
                    relaxation_loss += max_logit + torch.log(
                        torch.sum(torch.exp(high_sim_logits_with_baseline - max_logit))
                    )
                    valid_high_sim_pairs += high_sim_indices.numel()
            
            # Normalize by number of high-similarity pairs
            if valid_high_sim_pairs > 0:
                relaxation_loss = relaxation_loss / valid_high_sim_pairs
        
        # Dynamically adjust beta to prevent one term from dominating
        # This helps stabilize training
        effective_beta = self.beta
        if self.rounds_trained > 10 if hasattr(self, 'rounds_trained') else False:
            if relaxation_loss > 2.0 * scl_loss:
                effective_beta = self.beta * 0.5
            elif scl_loss > 2.0 * relaxation_loss:
                effective_beta = self.beta * 1.5
        
        # Combine losses with safeguards
        total_loss = scl_loss + effective_beta * relaxation_loss
        
        # Clip the loss to prevent explosion
        total_loss = torch.clamp(total_loss, 0.0, 10.0)
            
        return total_loss


class MultiLevelContrastiveLoss(nn.Module):
    """
    Multi-level contrastive learning loss that applies RCL across multiple network layers
    """
    def __init__(self, temperature=0.05, beta=1.0, lambda_threshold=0.7, layer_weights=None):
        super(MultiLevelContrastiveLoss, self).__init__()
        self.rcl = RelaxedContrastiveLoss(temperature, beta, lambda_threshold)
        
        # Default layer weights if none provided
        if layer_weights is None:
            self.layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights for 5 layers
        else:
            self.layer_weights = layer_weights
    
    def forward(self, features_dict, labels):
        """
        Apply RCL to multiple feature levels
        
        Args:
            features_dict: Dictionary of features from different layers
                          e.g., {'layer0': tensor, 'layer1': tensor, ...}
            labels: Class labels for the batch
        """
        total_loss = 0.0
        
        # Apply contrastive loss to each layer with corresponding weight
        for i, layer_name in enumerate([f'layer{j}' for j in range(len(self.layer_weights))]):
            if layer_name in features_dict:
                features = features_dict[layer_name]
                
                # For convolutional features, apply global average pooling
                if len(features.shape) > 2:
                    features = F.adaptive_avg_pool2d(features, 1)
                    features = features.view(features.size(0), -1)
                
                # L2 normalize features
                features = F.normalize(features, p=2, dim=1)
                
                # Apply RCL with corresponding weight
                layer_loss = self.rcl(features, labels)
                total_loss += self.layer_weights[i] * layer_loss
        
        return total_loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss between student and teacher models
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for balancing distillation and CE loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels=None, 
                student_features=None, teacher_features=None):
        """
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            labels: Ground truth labels (optional, for CE loss)
            student_features: Feature outputs from student model (optional)
            teacher_features: Feature outputs from teacher model (optional)
        """
        # Logit-based distillation
        temp = self.temperature
        soft_targets = F.softmax(teacher_logits / temp, dim=1)
        log_probs = F.log_softmax(student_logits / temp, dim=1)
        kd_loss = -(soft_targets * log_probs).sum(dim=1).mean() * (temp ** 2)
        
        # Feature-based distillation (optional)
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            student_features = F.normalize(student_features, dim=1)
            teacher_features = F.normalize(teacher_features, dim=1)
            feature_loss = (1 - (student_features * teacher_features).sum(dim=1)).mean()
        
        # Cross-entropy loss with true labels (optional)
        ce_loss = 0.0
        if labels is not None:
            ce_loss = self.ce_loss(student_logits, labels)
            
            # Combined loss with alpha weighting
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * (kd_loss + 0.5 * feature_loss)
        else:
            total_loss = kd_loss + 0.5 * feature_loss
        
        return total_loss


class HybridPersonalizationLoss(nn.Module):
    """
    Loss for the hybrid personalized head that combines:
    1. Cross-entropy for classification
    2. Knowledge distillation from global to personalized head
    3. L2 regularization on personalized parameters
    """
    def __init__(self, distillation_temp=3.0, distillation_weight=0.7, l2_reg_weight=0.001):
        super(HybridPersonalizationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.distillation_temp = distillation_temp
        self.distillation_weight = distillation_weight
        self.l2_reg_weight = l2_reg_weight
    
    def forward(self, outputs, targets, global_outputs=None, personalized_params=None):
        """
        Args:
            outputs: Output from personalized head (logits)
            targets: Target labels
            global_outputs: Output from global head (logits) for distillation
            personalized_params: Parameters of personalized head for L2 regularization
        """
        ce_loss = self.ce_loss(outputs, targets)
        
        # Knowledge distillation loss
        distillation_loss = 0.0
        if global_outputs is not None:
            temp = self.distillation_temp
            soft_targets = F.softmax(global_outputs / temp, dim=1)
            log_probs = F.log_softmax(outputs / temp, dim=1)
            distillation_loss = -(soft_targets * log_probs).sum(dim=1).mean() * (temp ** 2)
        
        # L2 regularization loss
        l2_loss = 0.0
        if personalized_params is not None:
            for param in personalized_params:
                l2_loss += torch.norm(param)**2
        
        # Combined loss
        total_loss = ce_loss + self.distillation_weight * distillation_loss + self.l2_reg_weight * l2_loss
        
        return total_loss, {
            'ce_loss': ce_loss,
            'distillation_loss': distillation_loss,
            'l2_loss': l2_loss
        }


class FedProxLoss(nn.Module):
    """
    FedProx regularization term for federated learning
    """
    def __init__(self, mu=0.01):
        super(FedProxLoss, self).__init__()
        self.mu = mu
    
    def forward(self, current_params, global_params, trust_score=1.0):
        """
        Compute FedProx regularization term
        
        Args:
            current_params: Parameters of the current model
            global_params: Parameters of the global model
            trust_score: Client trust score to adjust regularization (higher trust = lower reg)
        """
        proximal_term = 0.0
        for w, w_t in zip(current_params, global_params):
            if w.data.shape == w_t.data.shape:
                proximal_term += (w - w_t).norm(2)**2
        
        # Adjust mu based on trust score if provided
        effective_mu = self.mu
        if trust_score < 1.0:
            # Lower trust score means stronger regularization
            effective_mu = self.mu * (2.0 - trust_score)
            
        return effective_mu * proximal_term / 2


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) loss for continual learning
    """
    def __init__(self, lambda_ewc=0.4):
        super(EWCLoss, self).__init__()
        self.lambda_ewc = lambda_ewc
    
    def forward(self, model, fisher_info, optimal_params):
        """
        Compute EWC regularization loss
        
        Args:
            model: Current model
            fisher_info: Dictionary of Fisher information matrices {param_name: tensor}
            optimal_params: Dictionary of optimal parameters {param_name: tensor}
        """
        if not fisher_info or not optimal_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Calculate EWC loss for each parameter
        for name, param in model.named_parameters():
            if name in fisher_info and name in optimal_params:
                # Ensure tensors are on the same device
                fisher = fisher_info[name].to(param.device)
                optimal_param = optimal_params[name].to(param.device)
                
                # Calculate squared difference weighted by Fisher information
                loss += (fisher * (param - optimal_param).pow(2)).sum()
        
        return self.lambda_ewc * loss


class TrustBasedLoss(nn.Module):
    """
    Trust-based loss for federated learning that incorporates:
    1. FedProx regularization based on trust score
    2. EWC regularization for continual learning
    """
    def __init__(self, fedprox_mu=0.01, ewc_lambda=0.4):
        super(TrustBasedLoss, self).__init__()
        self.fedprox_loss = FedProxLoss(fedprox_mu)
        self.ewc_loss = EWCLoss(ewc_lambda)
    
    def forward(self, model, global_model, task_loss, fisher_information=None, 
                optimal_params=None, trust_score=1.0):
        """
        Combine task loss with FedProx and EWC regularization
        """
        # Compute FedProx term
        fedprox_loss = self.fedprox_loss(
            model.parameters(), 
            global_model.parameters(),
            trust_score
        )
        
        # Compute EWC loss
        ewc_loss = 0.0
        if fisher_information is not None and optimal_params is not None:
            ewc_loss = self.ewc_loss(model, fisher_information, optimal_params)
        
        # Combine losses
        total_loss = task_loss + fedprox_loss + ewc_loss
        
        return total_loss, {
            'task_loss': task_loss,
            'fedprox_loss': fedprox_loss,
            'ewc_loss': ewc_loss
        }


def compute_personalized_loss(output, target, model=None, reduction='mean'):
    """
    Compute personalized loss with class weighting for imbalanced data
    """
    if isinstance(output, dict):
        if "personalized_logit" in output:
            logits = output["personalized_logit"]
        else:
            logits = output["logit"] if "logit" in output else output
    else:
        logits = output
        
    # Compute class weights based on frequency in the batch
    class_counts = torch.bincount(target)
    class_weights = 1.0 / torch.clamp(class_counts[target], min=1.0)
    class_weights = class_weights / class_weights.mean()
    
    # Apply weighted cross-entropy loss
    loss = F.cross_entropy(logits, target, reduction='none')
    weighted_loss = loss * class_weights
    
    if reduction == 'mean':
        loss = weighted_loss.mean()
    elif reduction == 'sum':
        loss = weighted_loss.sum()
    else:
        loss = weighted_loss
    
    # Add L2 regularization for personalized head if available
    if model is not None and hasattr(model, 'personalized_head'):
        l2_reg = 0
        for param in model.personalized_head.parameters():
            l2_reg += torch.norm(param)
        
        loss += 0.005 * l2_reg
    
    return loss


def KL_u_p_loss(outputs):
    """
    Compute KL divergence between predictions and uniform distribution
    to encourage exploration and feature diversity
    """
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = torch.autograd.Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduction='none').sum(dim=1)
    return instance_losses


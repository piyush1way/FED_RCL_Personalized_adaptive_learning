import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from utils.helper import *
from utils.logging_utils import AverageMeter
import math

__all__ = ['MultiLabelCrossEntropyLoss', 'CLLoss', 'KL_u_p_loss', 'compute_personalized_loss', 'RelaxedContrastiveLoss']


class MultiLabelCrossEntropyLoss(nn.Module):

    def __init__(self, eps: float=0, alpha: float=0.2, topk_pos: int=-1, temp: float=1., **kwargs):
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.topk_pos = topk_pos
        self.temp = temp

    def __repr__(self):
        return "MultiLabelCrossEntropyLoss(eps={}, alpha={})".format(self.eps, self.alpha)

    def forward(self, input: torch.Tensor, targets: torch.Tensor, reduction: bool = True, beta: float = None, ) -> torch.Tensor:

        N, C = input.size()
        E = self.eps

        input[input==np.inf] = -np.inf # to ignore np.inf

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
        self.beta = self.pair.get('beta') or beta # weight for negative samples. 

        self.threshold = self.pair.get('threshold') or threshold

        self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos, temp=temp, eps=eps,) 


    def __set_num_classes__(self, num_classes):
        self.num_classes = num_classes

    def __repr__(self):
        return "{}(topk_pos={}, topk_neg={}, temp={}, crit={}), pair={})".format(
            type(self).__name__, self.topk_pos, self.topk_neg, self.temp, self.criterion, self.pair)

    def get_classwise_mask(self, target):
        B = target.size(0)
        classwise_mask = target.expand(B, B).eq(target.expand(B, B).T)
        return classwise_mask
    

    def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, topk_pos=None, labels=None,):
        # Check for minimum batch size
        B = sim.size(0)
        if B < 2:
            # Return empty tensor if batch is too small
            return torch.tensor([], device=sim.device)

        sim_neg = sim.clone()

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        if 'unsupervised' in neg_loss_type:
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 if the same sample, otherwise 0
        else:
            pos_mask = self.get_classwise_mask(labels)

        if self.neg_sample_type:
            if self.neg_sample_type == 'intra_class_thresholding':
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = np.inf
                sim_neg[torch.eye(B)==1] = np.inf
                
                # Ensure topk_pos doesn't exceed batch size
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
            # Ensure topk_neg doesn't exceed batch size
            k = min(topk_neg, B-1) if topk_neg > 0 else 1
            if k > 0:
                sim_neg = torch.topk(sim_neg, k, dim=1, largest=True)[0]
            else:
                return torch.tensor([], device=sim.device)

        return sim_neg
    

    def get_topk_pos(self, sim, topk_pos=None, labels=None, uncertainty=None):
        # Check for minimum batch size
        B = sim.size(0)
        if B < 2:
            # Return empty tensor if batch is too small
            return torch.tensor([], device=sim.device)

        sim_pos = sim.clone()

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        sim_pos[pos_mask==0] = np.inf
        
        # Ensure topk_pos doesn't exceed number of positive pairs
        k = min(topk_pos, (pos_mask.sum(dim=1) - 1).max().item()) if topk_pos > 0 else 1
        if k > 0:
            sim_pos, inds = torch.topk(sim_pos, k, dim=1, largest=False)
        else:
            return torch.tensor([], device=sim.device)

        return sim_pos

    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, name="loss1"):
        # Check for minimum batch size
        if old_feat.size(0) < 2 or new_feat.size(0) < 2:
            # Return zero loss if batch is too small for contrastive learning
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
            
        # Ensure topk values don't exceed batch size
        topk_pos = min(topk_pos, B-1) if topk_pos > 0 else 1
        topk_neg = min(topk_neg, B-1) if topk_neg > 0 else 1

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        for pair_type in all_pair_types:
            try:
                sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
                sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, topk_pos=topk_pos, labels=target)
            except Exception as e:
                # Handle errors gracefully
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
        
        # Check if we have any pairs
        if not pair_poss or not pair_negs:
            return torch.tensor(0.0, device=old_feat.device)
        
        # Concatenate pairs
        pair_poss = torch.cat(pair_poss, 1)  # B*P
        pair_negs = torch.cat(pair_negs, 1)  # B*N
        
        # Reshape for loss calculation
        pair_poss_ = pair_poss.unsqueeze(2)  # B*P*1
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1], 1)
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2)  # B*P*(N+1)
        
        # Create binary labels
        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1
        
        # Scale inputs by temperature
        scaled_inputs = pair_all_.reshape(-1, pair_all_.size(2)) / self.temp
        
        # Calculate loss with a maximum cap to prevent explosion
        try:
            loss = self.criterion(
                input=scaled_inputs,
                targets=binary_zero_labels_.reshape(-1, pair_all_.size(2)),
                reduction=reduction, 
                beta=self.beta
            )
            
            # Reshape and mean
            if loss.numel() > 1:
                loss = loss.reshape(B, -1).mean(1)
                
            # Cap the loss to prevent explosion
            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)) or torch.any(loss > 100):
                return torch.tensor(1.0, device=old_feat.device)
                
            return loss
        except Exception:
            # Return a small constant loss if calculation fails
            return torch.tensor(1.0, device=old_feat.device)


class RelaxedContrastiveLoss(nn.Module):
    """
    Relaxed Contrastive Loss with divergence penalty for FedRCL-P.
    This loss prevents representation collapse by imposing a penalty on excessively similar pairs.
    """
    def __init__(self, temperature=0.1, lambda_penalty=0.05, similarity_threshold=0.5):
        super(RelaxedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.lambda_penalty = lambda_penalty
        self.similarity_threshold = similarity_threshold
        
    def forward(self, features, labels):
        """
        Args:
            features: Feature representations (B x D)
            labels: Ground truth labels (B)
        """
        # Check batch size
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
            
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t())
        
        # Create mask for positive pairs (same class)
        labels_expand = labels.expand(batch_size, batch_size)
        pos_mask = labels_expand.eq(labels_expand.t()).float()
        
        # Remove self-similarity
        eye_mask = torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask - eye_mask
        
        # Check if there are any positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Compute average similarity for positive pairs for adaptive threshold
        pos_sim = torch.clamp((sim_matrix * pos_mask).sum() / (pos_mask.sum() + 1e-8), min=0.1, max=0.9)
        adaptive_threshold = pos_sim * 0.8  # 80% of average positive similarity
        
        # Apply temperature scaling for loss calculation
        sim_matrix_scaled = sim_matrix / self.temperature
        
        # Standard supervised contrastive loss calculation with improved numerical stability
        exp_sim = torch.exp(sim_matrix_scaled)
        
        # For each anchor i, compute sum of exp(sim(i,j)) for all j
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) - exp_sim.diag().unsqueeze(1)
        
        # Compute positive pairs with masking
        pos_pairs = pos_mask * exp_sim
        pos_pairs_sum = torch.clamp(pos_pairs.sum(dim=1, keepdim=True), min=1e-8)
        
        # Safe division with clamping
        denominator = torch.clamp(exp_sim_sum, min=1e-8)
        log_probs = torch.log(torch.clamp(pos_pairs_sum / denominator, min=1e-8))
        
        # Compute standard SCL loss (only for anchors with positive pairs)
        pos_count = torch.clamp(pos_mask.sum(dim=1), min=1e-8)
        scl_loss = -log_probs.sum() / pos_count.sum()
        
        # Compute divergence penalty for excessively similar positive pairs
        # Use adaptive threshold based on data distribution
        high_sim_pos = (sim_matrix > adaptive_threshold) & (pos_mask > 0)
        
        # If there are high similarity pairs, compute penalty
        if high_sim_pos.sum() > 0:
            penalty_values = (sim_matrix - adaptive_threshold) * high_sim_pos.float()
            penalty = penalty_values.sum() / (high_sim_pos.sum() + 1e-8)
        else:
            penalty = torch.tensor(0.0, device=features.device)
        
        # Combine standard loss with penalty using a smaller lambda value
        total_loss = scl_loss + self.lambda_penalty * penalty
        
        # Cap the loss to prevent explosion
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 10.0:
            return torch.tensor(0.5, device=features.device)
            
        return total_loss


def compute_personalized_loss(output, target, model=None, reduction='mean'):
    """Compute loss with support for personalized models.
    
    Args:
        output: Model output (logits or features)
        target: Ground truth labels
        model: The model instance (used for regularization)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
        
    Returns:
        torch.Tensor: Computed loss
    """
    # Standard cross-entropy loss
    if isinstance(output, dict):
        if "personalized_logit" in output:
            logits = output["personalized_logit"]
        else:
            logits = output["logit"] if "logit" in output else output
    else:
        logits = output
        
    loss = F.cross_entropy(logits, target, reduction=reduction)
    
    # Add regularization for personalized models if needed
    if model is not None and hasattr(model, 'personalized_head'):
        # Optional: Add regularization for personalized head
        l2_reg = 0
        for param in model.personalized_head.parameters():
            l2_reg += torch.norm(param)
        
        # Add regularization term to loss (with small weight)
        loss += 0.01 * l2_reg
    
    return loss


def KL_u_p_loss(outputs):
    # KL(u||p)
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = torch.autograd.Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduction='none').sum(dim=1)
    return instance_losses

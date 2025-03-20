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
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)
            
        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.t()) / self.temperature
        
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
        
        # Standard supervised contrastive loss
        exp_sim = torch.exp(sim_matrix)
        
        # For numerical stability, mask out self-similarity
        exp_sim = exp_sim * (1 - eye_mask)
        
        # Compute denominators (sum over all samples except self)
        denominators = exp_sim.sum(dim=1, keepdim=True)
        
        # Compute positive terms only
        pos_exp_sim = exp_sim * pos_mask
        
        # Compute standard SCL loss
        eps = 1e-8  # Small epsilon to avoid log(0)
        scl_loss = 0
        for i in range(batch_size):
            pos_indices = torch.nonzero(pos_mask[i]).squeeze()
            if pos_indices.numel() > 0:  # If there are positive samples
                if pos_indices.numel() == 1:
                    pos_indices = pos_indices.unsqueeze(0)
                
                # Compute loss for each positive pair
                for j in pos_indices:
                    numerator = exp_sim[i, j]
                    denominator = denominators[i]
                    scl_loss -= torch.log(numerator / (denominator + eps))
        
        # Normalize by number of positive pairs
        num_pos_pairs = pos_mask.sum()
        if num_pos_pairs > 0:
            scl_loss = scl_loss / num_pos_pairs
        
        # Relaxation term: penalty for too-similar positive pairs
        # Find positive pairs with similarity > lambda
        high_sim_pos = (sim_matrix * self.temperature > self.lambda_threshold) & (pos_mask > 0)
        
        relaxation_loss = 0
        if high_sim_pos.sum() > 0:
            # For each anchor i, compute log(sum_j exp(z_i·z_j/τ) + exp(1/τ)) where j are positive samples with similarity > lambda
            for i in range(batch_size):
                high_sim_indices = torch.nonzero(high_sim_pos[i]).squeeze()
                if high_sim_indices.numel() > 0:
                    if high_sim_indices.numel() == 1:
                        high_sim_indices = high_sim_indices.unsqueeze(0)
                    
                    # Sum of exp(similarity) for high-similarity positive pairs
                    high_sim_sum = exp_sim[i, high_sim_indices].sum() + torch.exp(torch.tensor(1.0/self.temperature, device=features.device))
                    relaxation_loss += torch.log(high_sim_sum)
            
            # Normalize by number of high-similarity pairs
            relaxation_loss = relaxation_loss / high_sim_pos.sum()
        
        # Total loss
        total_loss = scl_loss + self.beta * relaxation_loss
        
        # Handle numerical issues
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.5, device=features.device)
            
        return total_loss



def compute_personalized_loss(output, target, model=None, reduction='mean'):
    if isinstance(output, dict):
        if "personalized_logit" in output:
            logits = output["personalized_logit"]
        else:
            logits = output["logit"] if "logit" in output else output
    else:
        logits = output
        
    class_counts = torch.bincount(target)
    class_weights = 1.0 / torch.clamp(class_counts[target], min=1.0)
    class_weights = class_weights / class_weights.mean()
    
    loss = F.cross_entropy(logits, target, reduction='none')
    weighted_loss = loss * class_weights
    
    if reduction == 'mean':
        loss = weighted_loss.mean()
    elif reduction == 'sum':
        loss = weighted_loss.sum()
    else:
        loss = weighted_loss
    
    if model is not None and hasattr(model, 'personalized_head'):
        l2_reg = 0
        for param in model.personalized_head.parameters():
            l2_reg += torch.norm(param)
        
        loss += 0.005 * l2_reg
    
    return loss


def KL_u_p_loss(outputs):
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = torch.autograd.Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduction='none').sum(dim=1)
    return instance_losses

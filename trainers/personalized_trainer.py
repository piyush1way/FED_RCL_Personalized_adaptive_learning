import logging
from trainers.base_trainer import BaseTrainer
from trainers.build import TRAINER_REGISTRY
from utils.metrics import evaluate, track_trust_scores, evaluate_personalization_benefits

logger = logging.getLogger(__name__)

@TRAINER_REGISTRY.register()
class PersonalizedTrainer(BaseTrainer):
    """Trainer with personalization support for FedRCL"""
    
    def __init__(self, args, model, trainset, testset, clients, server, evaler):
        super().__init__(args, model, trainset, testset, clients, server, evaler)
        self.enable_personalization = getattr(args.trainer, "personalization", {}).get("enable", True)
        
    def evaluate(self, round_idx):
        """Evaluate both global and personalized models"""
        metrics = super().evaluate(round_idx)
        
        # Track trust scores if available
        if hasattr(self.server, 'trust_scores') and len(self.server.trust_scores) > 0:
            trust_stats = track_trust_scores(self)
            metrics.update({"trust_stats": trust_stats})
            
        # Evaluate personalization benefits
        if self.enable_personalization:
            personalization_metrics = evaluate_personalization_benefits(
                self.args, self.model, self.testloader, self.device
            )
            metrics.update({"personalization": personalization_metrics})
            
        return metrics
    
    def select_clients(self, round_idx):
        """Select clients with trust-based selection if available"""
        if hasattr(self.server, 'select_clients') and callable(self.server.select_clients):
            # Use server's trust-based client selection if available
            num_clients = max(1, int(self.args.trainer.participation_rate * len(self.clients)))
            return self.server.select_clients(list(self.clients.keys()), num_clients)
        else:
            # Fall back to random selection
            return super().select_clients(round_idx)

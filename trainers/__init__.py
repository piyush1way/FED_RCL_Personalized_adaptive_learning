# trainers/__init__.py
from trainers.build import TRAINER_REGISTRY, get_trainer_type
from trainers.base_trainer import BaseTrainer

# Register any additional trainer classes here if needed
from .personalized_trainer import PersonalizedTrainer

__all__ = ['BaseTrainer', 'PersonalizedTrainer', 'TRAINER_REGISTRY', 'get_trainer_type']

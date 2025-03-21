import os
from pathlib import Path
import time
import json
import numpy as np
import copy

import torch
import wandb
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from datasets.build import build_dataset, build_datasets
from models.build import build_encoder
from servers.build import get_server_type, build_server
from clients.build import get_client_type
from evalers.build import get_evaler_type
from trainers.build import get_trainer_type
from utils.metrics import track_trust_scores, evaluate_personalization_benefits
from utils.helper import save_dict_to_json

from utils import initalize_random_seed

import hydra
from omegaconf import DictConfig
import omegaconf
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(asctime)s[%(name)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

wandb.require("service")


@hydra.main(version_base=None, config_path="configs", config_name="fedrcl_p")
def main(args: DictConfig) -> None:
    start_time = time.time()
    
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.backends.cuda.max_split_size_mb = 512
            device = torch.device("cuda:0")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f"Multiprocessing initialization failed: {e}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    args.log_dir = Path(args.log_dir)
    exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
    args.log_dir = args.log_dir / args.dataset.name / exp_name
    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment Name: {exp_name}")

    if args.wandb:
        wandb.init(
            entity='federated_learning',
            project=args.project,
            group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
            job_type=exp_name,
            dir=args.log_dir,
        )
        wandb.run.name = exp_name
        wandb.config.update(omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

    initalize_random_seed(args)

    # Build model with personalization support
    model = build_encoder(args)
    
    # Check if model supports personalization
    personalization_enabled = getattr(args.client.personalization, "enable", False)
    if personalization_enabled:
        if hasattr(model, 'enable_personalized_mode'):
            model.enable_personalized_mode()
            logger.info("Personalized mode enabled for the model")
        else:
            logger.warning("Model does not support personalization. Using standard model.")
    
    # Get client and server types
    client_type = get_client_type(args)
    server_type = get_server_type(args)
    
    # Create server instance
    # server = server_type(args)
    # server.setup(model)  # Initialize the server with the model
    # Create server instance
    server = build_server(args)  # Use build_server instead of direct instantiation
    server.setup(model)
    
    # Build datasets with balanced subset sharing if enabled
    if hasattr(args.split, 'share_balanced_subset') and args.split.share_balanced_subset:
        logger.info(f"Creating balanced subset to share across clients")
        datasets = build_datasets(args)
    else:
        datasets = build_datasets(args)
    
    # Create client instances
    client_instances = {}
    for client_idx in range(args.trainer.num_clients):
        client_instances[client_idx] = client_type(args, client_idx, copy.deepcopy(model))
    
    # Get evaler and trainer types
    evaler_type = get_evaler_type(args)
    trainer_type = get_trainer_type(args)

    logger.info(f"Model: {args.model.name}, Pretrained={args.model.pretrained}")
    logger.info(f"Client Type: {client_type.__name__}")
    logger.info(f"Server Type: {server.__class__.__name__}")
    
    # Log configuration details
    cyclical_lr_enabled = getattr(args.client, "cyclical_lr", False)
    fedprox_enabled = getattr(args.client, "fedprox", False)
    distillation_enabled = getattr(args.client, "distillation", False)
    trust_filtering_enabled = getattr(args.client.trust_filtering, "enable", False) if hasattr(args.client, "trust_filtering") else False
    multi_level_rcl = getattr(args.client, "multi_level_rcl", False)
    ewc_enabled = getattr(args.client, "ewc_enabled", False)
    
    logger.info(f"Personalization: {personalization_enabled}")
    logger.info(f"Cyclical Learning Rate: {cyclical_lr_enabled}")
    logger.info(f"FedProx Regularization: {fedprox_enabled}")
    logger.info(f"Knowledge Distillation: {distillation_enabled}")
    logger.info(f"Trust Filtering: {trust_filtering_enabled}")
    logger.info(f"Multi-Level Contrastive Learning: {multi_level_rcl}")
    logger.info(f"EWC Regularization: {ewc_enabled}")
    logger.info(f"Data Split Mode: {args.split.mode}, Alpha: {args.split.alpha}")

    # Create evaler
    evaler = evaler_type(args)
    logger.info(f"=> Creating evaler '{evaler_type.__name__}'")
    
    # Create trainer
    logger.info(f"=> Creating trainer '{trainer_type.__name__}'")
    trainer = trainer_type(
        args=args,
        model=model,
        trainset=datasets['train'],
        testset=datasets['test'],
        clients=client_instances,  # Pass the dictionary of client instances
        server=server,             # Pass the server instance
        evaler=evaler
    )

    # Track metrics during training
    training_metrics = {
        "global_acc": [],
        "personalized_acc": [],
        "trust_scores": [],
        "client_participation": [],
        "training_loss": [],
        "rcl_loss": [],
        "distillation_loss": [],
        "fedprox_loss": [],
        "ewc_loss": []
    }

    try:
        # Train the model
        trained_model, metrics = trainer.train()
        if metrics:
            training_metrics.update(metrics)
    except RuntimeError as e:
        logger.error(f"Training Failed: {e}")
        raise e
    
    logger.info("Training completed. Running final evaluation...")
    final_results = trainer.evaluate(round_idx=args.trainer.global_rounds-1)
    
    # Log final results
    if args.wandb:
        wandb.log({
            "final_acc": final_results["acc"],
            "final_personalized_acc": final_results.get("acc_personalized", final_results["acc"]),
            "training_time": time.time() - start_time
        })
    
    logger.info(f"Final Global Model Accuracy: {final_results['acc']:.2f}%")
    if "acc_personalized" in final_results:
        logger.info(f"Final Personalized Model Accuracy: {final_results['acc_personalized']:.2f}%")
        improvement = final_results['acc_personalized'] - final_results['acc']
        logger.info(f"Personalization Improvement: {improvement:.2f}%")
    
    # Save final model
    if hasattr(trainer, 'save_model'):
        trainer.save_model(suffix="final")
    
    # Evaluate personalization benefits if enabled
    if personalization_enabled:
        logger.info("Evaluating personalization benefits...")
        personalization_metrics = evaluate_personalization_benefits(
            args, model, trainer.testloader, device
        )
        
        # Save personalization metrics
        metrics_path = args.log_dir / "personalization_metrics.json"
        save_dict_to_json(personalization_metrics, metrics_path)
        
        if args.wandb:
            wandb.log({"personalization_metrics": personalization_metrics})
    
    # Save training metrics
    metrics_path = args.log_dir / "training_metrics.json"
    save_dict_to_json(training_metrics, metrics_path)
    
    # Log novelty effectiveness
    if multi_level_rcl and "acc_personalized" in final_results:
        logger.info("Novelty Effectiveness:")
        if "acc_personalized" in final_results and "acc" in final_results:
            improvement = final_results['acc_personalized'] - final_results['acc']
            logger.info(f"1. Hybrid Personalized Head: {improvement:.2f}% accuracy improvement")
        
        if hasattr(trainer.server, 'trust_scores') and len(trainer.server.trust_scores) > 0:
            trust_stats = track_trust_scores(trainer)
            logger.info(f"2. Trust-Based Adaptive LR: Mean trust score {trust_stats['mean_trust']:.4f}")
            logger.info(f"3. Soft Trust-Based Filtering: {trust_stats['trusted_clients']}/{trust_stats['total_clients']} trusted clients")
        
        if "bop" in personalization_metrics:
            logger.info(f"4. Multi-Level Contrastive Learning: {personalization_metrics['bop']:.4f} benefit of personalization")
    
    logger.info(f"Experiment completed successfully! Total time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()

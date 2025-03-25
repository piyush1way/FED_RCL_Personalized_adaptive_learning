import os
from pathlib import Path
import time
import json
import numpy as np
import copy
import gc

import torch
from torch.utils.data import DataLoader
import wandb

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
    """Main entry point for federated training"""
    start_time = time.time()
    
    # Configure CUDA for better memory management
    if torch.cuda.is_available():
        try:
            # Clear CUDA cache and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            
            # Apply memory split size if specified in config
            if hasattr(args, 'memory') and hasattr(args.memory, 'max_split_size_mb'):
                max_split_size_mb = args.memory.max_split_size_mb
            else:
                # Default to 64MB for Kaggle (smaller than previous 128MB)
                max_split_size_mb = 64
                
            # Set environment variables for PyTorch memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb},garbage_collection_threshold:0.6'
            logger.info(f"Set CUDA max_split_size_mb to {max_split_size_mb}")
                
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)  # Limit to 80% instead of default 95%
            
            # Set other performance options
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
            torch.backends.cudnn.allow_tf32 = True
            
            device = torch.device("cuda:0")
            # Print GPU info for debugging
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Capability: {torch.cuda.get_device_capability()}")
            
            # Auto-adjust batch size based on GPU memory
            if not hasattr(args, 'batch_size') or args.batch_size > 16:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb < 12:  # For smaller GPUs like K80, P100, T4
                    args.batch_size = 8
                    logger.info(f"Auto-adjusted batch size to {args.batch_size} for {gpu_memory_gb:.1f} GB GPU")
            
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

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
    model = model.to(device)
    
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
    
    # Create server instance with memory-efficient approach
    server = build_server(args)
    try:
        # Move model to CPU temporarily for setup to avoid OOM
        model_device = next(model.parameters()).device
        model = model.to('cpu')
        server.setup(model)  # Initialize the server with the model
        model = model.to(model_device)  # Move back
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during server setup. Using fallback approach.")
            # Clear cache and retry
            torch.cuda.empty_cache()
            gc.collect()
            # Try again with more conservative approach
            try:
                # Create new server and setup
                server = build_server(args)
                server.setup(model)
            except Exception as nested_e:
                logger.error(f"Server setup failed: {nested_e}")
                raise nested_e
        else:
            raise e
    
    # Build datasets with balanced subset sharing if enabled
    if hasattr(args.split, 'share_balanced_subset') and args.split.share_balanced_subset:
        logger.info(f"Creating balanced subset to share across clients")
        datasets = build_datasets(args)
    else:
        datasets = build_datasets(args)
    
    # Create client instances with memory-efficient approach
    client_instances = {}
    for client_idx in range(args.trainer.num_clients):
        try:
            # Create client with state dict rather than full model copy
            client = client_type(args, client_idx, None)  # Pass None instead of model copy
            client.model = copy.deepcopy(model)  # Set model directly
            client_instances[client_idx] = client
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM creating client {client_idx}. Using alternative approach.")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Alternative approach: create client without model, will be set during training
                client = client_type(args, client_idx, None)
                client_instances[client_idx] = client
            else:
                raise e
    
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
        clients=client_instances,
        server=server,
        evaler=evaler
    )

    # Move model to device
    trainer.device = device
    trainer.model = trainer.model.to(device)
    
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
        # Run garbage collection before training
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train the model
        trained_model, metrics = trainer.train()
        if metrics:
            training_metrics.update(metrics)
    except RuntimeError as e:
        logger.error(f"Training Failed: {e}")
        # Try to recover from CUDA OOM by clearing cache
        if 'CUDA out of memory' in str(e):
            logger.warning("CUDA OOM detected - attempting recovery")
            torch.cuda.empty_cache()
            gc.collect()
            # Try to continue with reduced batch size if possible
            if hasattr(args, 'batch_size') and args.batch_size > 8:
                old_batch_size = args.batch_size
                args.batch_size = args.batch_size // 2
                logger.info(f"Reduced batch size to {args.batch_size}")
                
                # Reset model to clear gradients
                if hasattr(trainer, 'model'):
                    for param in trainer.model.parameters():
                        if param.grad is not None:
                            param.grad = None
                
                # Try one more time
                try:
                    trained_model, metrics = trainer.train()
                    if metrics:
                        training_metrics.update(metrics)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    raise retry_e
            else:
                raise e
        else:
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
        try:
            trainer.save_model(suffix="final")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            # Fallback to simpler save
            try:
                save_path = args.log_dir / "final_model.pt"
                torch.save(trained_model.state_dict(), save_path)
                logger.info(f"Saved model state dict to {save_path}")
            except Exception as nested_e:
                logger.error(f"Fallback save failed: {nested_e}")
    
    # Evaluate personalization benefits if enabled
    if personalization_enabled:
        logger.info("Evaluating personalization benefits...")
        try:
            personalization_metrics = evaluate_personalization_benefits(
                args, model, trainer.testloader, device
            )
            
            # Save personalization metrics
            metrics_path = args.log_dir / "personalization_metrics.json"
            save_dict_to_json(personalization_metrics, metrics_path)
            
            if args.wandb:
                wandb.log({"personalization_metrics": personalization_metrics})
        except Exception as e:
            logger.error(f"Error evaluating personalization benefits: {e}")
            personalization_metrics = {"error": str(e)}
    
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
    
    return trained_model, training_metrics

if __name__ == "__main__":
    main()

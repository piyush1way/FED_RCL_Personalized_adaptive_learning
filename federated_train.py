# import os
# from pathlib import Path

# import torch
# import wandb
# from torch.multiprocessing import set_start_method
# from torch.utils.data import DataLoader

# from datasets.build import build_dataset, build_datasets
# from models.build import build_encoder
# from servers.build import get_server_type, build_server
# from clients.build import get_client_type
# from evalers.build import get_evaler_type
# from trainers.build import get_trainer_type

# from utils import initalize_random_seed

# import hydra
# from omegaconf import DictConfig
# import omegaconf
# import coloredlogs, logging

# logger = logging.getLogger(__name__)
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

# wandb.require("service")


# @hydra.main(version_base=None, config_path="configs", config_name="config")
# def main(args: DictConfig) -> None:
#     # ======= Fix CUDA Memory Issue =======
#     if torch.cuda.is_available():
#         try:
#             torch.cuda.empty_cache()  # Clear unused memory
#             torch.cuda.ipc_collect()  # Collect fragmented memory
#             torch.backends.cuda.max_split_size_mb = 512  # Limit memory splits (if needed)
#             device = torch.device("cuda:0")
#         except Exception as e:
#             logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
#             device = torch.device("cpu")
#     else:
#         device = torch.device("cpu")

#     print(f" Using device: {device}")

#     # Set multiprocessing strategy
#     try:
#         torch.multiprocessing.set_sharing_strategy('file_system')
#         set_start_method('spawn', force=True)
#     except RuntimeError as e:
#         logger.warning(f"Multiprocessing initialization failed: {e}")

#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#     # ======= Setup Logging Directory =======
#     args.log_dir = Path(args.log_dir)
#     exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
#     args.log_dir = args.log_dir / args.dataset.name / exp_name
#     if not args.log_dir.exists():
#         args.log_dir.mkdir(parents=True, exist_ok=True)

#     print(f" Experiment Name: {exp_name}")

#     # ======= Initialize Wandb =======
#     if args.wandb:
#         wandb.init(
#             entity='federated_learning',
#             project=args.project,
#             group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
#             job_type=exp_name,
#             dir=args.log_dir,
#         )
#         wandb.run.name = exp_name
#         wandb.config.update(omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

#     # ======= Set Random Seed =======
#     initalize_random_seed(args)

#     # ======= Build Components =======
#     model = build_encoder(args)
#     client_type = get_client_type(args)
#     server = build_server(args)
#     datasets = build_datasets(args)
#     evaler_type = get_evaler_type(args)
#     trainer_type = get_trainer_type(args)

#     # ======= Debugging Logs =======
#     logger.info(f" Model: {args.model.name}, Pretrained={args.model.pretrained}")
#     logger.info(f" Client Type: {client_type.__name__}")
#     logger.info(f" Server Type: {server.__class__.__name__}")

#     # ======= Initialize Trainer =======
#     trainer = trainer_type(
#         model=model,
#         client_type=client_type,
#         server=server,
#         evaler_type=evaler_type,
#         datasets=datasets,
#         device=device,
#         args=args,
#         config=None,
#     )

#     # ======= Start Training =======
#     try:
#         trainer.train()
#     except RuntimeError as e:
#         logger.error(f" Training Failed: {e}")
#         raise e


# if __name__ == '__main__':
#     main()
import os
from pathlib import Path

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
from utils.metrics import track_trust_scores

from utils import initalize_random_seed

import hydra
from omegaconf import DictConfig
import omegaconf
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

wandb.require("service")


@hydra.main(version_base=None, config_path="configs", config_name="fedrcl_p")
def main(args: DictConfig) -> None:
    # ======= Fix CUDA Memory Issue =======
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()  # Clear unused memory
            torch.cuda.ipc_collect()  # Collect fragmented memory
            torch.backends.cuda.max_split_size_mb = 512  # Limit memory splits (if needed)
            device = torch.device("cuda:0")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f" Using device: {device}")

    # Set multiprocessing strategy
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f"Multiprocessing initialization failed: {e}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # ======= Setup Logging Directory =======
    args.log_dir = Path(args.log_dir)
    exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
    args.log_dir = args.log_dir / args.dataset.name / exp_name
    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f" Experiment Name: {exp_name}")

    # ======= Initialize Wandb =======
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

    # ======= Set Random Seed =======
    initalize_random_seed(args)

    # ======= Build Components =======
    model = build_encoder(args)
    client_type = get_client_type(args)
    server = build_server(args)
    datasets = build_datasets(args)
    evaler_type = get_evaler_type(args)
    trainer_type = get_trainer_type(args)

    # ======= Debugging Logs =======
    logger.info(f" Model: {args.model.name}, Pretrained={args.model.pretrained}")
    logger.info(f" Client Type: {client_type.__name__}")
    logger.info(f" Server Type: {server.__class__.__name__}")
    
    # Log personalization settings
    personalization_enabled = getattr(args.client.personalization, "enable", False)
    adaptive_lr_enabled = getattr(args.client.adaptive_lr, "enable", False)
    trust_filtering_enabled = getattr(args.client.trust_filtering, "enable", False)
    
    logger.info(f" Personalization: {personalization_enabled}")
    logger.info(f" Adaptive Learning Rate: {adaptive_lr_enabled}")
    logger.info(f" Trust Filtering: {trust_filtering_enabled}")

    # ======= Initialize Trainer =======
    trainer = trainer_type(
        model=model,
        client_type=client_type,
        server=server,
        evaler_type=evaler_type,
        datasets=datasets,
        device=device,
        args=args,
        config=None,
    )

    # ======= Start Training =======
    try:
        trainer.train()
    except RuntimeError as e:
        logger.error(f" Training Failed: {e}")
        raise e
    
    # ======= Final Evaluation =======
    logger.info(" Training completed. Running final evaluation...")
    final_results = trainer.evaluate(epoch=args.trainer.global_rounds)
    
    if args.wandb:
        wandb.log({
            "final_acc": final_results["acc"],
            "final_personalized_acc": final_results.get("acc_personalized", final_results["acc"])
        })
    
    logger.info(f" Final Global Model Accuracy: {final_results['acc']:.2f}%")
    if "acc_personalized" in final_results:
        logger.info(f" Final Personalized Model Accuracy: {final_results['acc_personalized']:.2f}%")
    
    # Save the final model
    trainer.save_model(epoch=args.trainer.global_rounds-1, suffix="final")
    
    logger.info(" Experiment completed successfully!")


if __name__ == '__main__':
    main()

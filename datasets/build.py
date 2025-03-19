from utils.registry import Registry
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
import yaml
import torch
import numpy as np
from torch.utils.data import Subset, Dataset
from utils.data import create_balanced_subset, share_balanced_data, DatasetSplit

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""

DATASET_REGISTRY.register(CIFAR10)
DATASET_REGISTRY.register(CIFAR100)

__all__ = ['build_dataset', 'build_datasets']

def get_transform(args, train, config):
    if 'leaf_femnist' in args.dataset.name:
        transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            ToTensor()])
    elif 'leaf_celeba' in args.dataset.name:
        transform = transforms.Compose([
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        color_jitter = transforms.ColorJitter(0.4 * 1, 0.4 * 1, 0.4 * 1, 0.1 * 1)
        normalize = transforms.Normalize(config['mean'],
                                         config['std'])
        imsize = config['imsize']
        if train:
            transform = transforms.Compose(
                [transforms.RandomRotation(10),
                 transforms.RandomCrop(imsize, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize
                 ])
        else:
            transform = transforms.Compose(
                [transforms.CenterCrop(imsize),
                 transforms.ToTensor(),
                 normalize])
    return transform

def build_dataset(args, train=True):
    if args.verbose and train == True:
        print(DATASET_REGISTRY)

    download = args.dataset.download if args.dataset.get('download') else False

    with open('datasets/configs.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)[args.dataset.name]

    transform = get_transform(args, train, dataset_config)

    dataset = DATASET_REGISTRY.get(args.dataset.name)(root=args.dataset.path, download=download, train=train, transform=transform) if len(args.dataset.path) > 0 else None

    return dataset

def split_data_dirichlet(dataset, num_clients, alpha):
    labels = np.array([target for _, target in dataset])
    num_classes = len(np.unique(labels))
    
    client_indices = [[] for _ in range(num_clients)]
    
    class_indices = [np.where(labels == class_id)[0] for class_id in range(num_classes)]
    
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        proportions = np.array([p * len(indices) for p in proportions])
        proportions = np.around(proportions).astype(int)
        
        diff = len(indices) - np.sum(proportions)
        proportions[0] += diff
        
        index = 0
        for client_id, prop in enumerate(proportions):
            client_indices[client_id].extend(indices[index:index + prop].tolist())
            index += prop
    
    client_datasets = {}
    for client_id, indices in enumerate(client_indices):
        if len(indices) > 0:
            client_datasets[client_id] = Subset(dataset, indices)
    
    return client_datasets

def build_datasets(args):
    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)
    
    if hasattr(args, 'trainer') and hasattr(args.trainer, 'num_clients'):
        num_clients = args.trainer.num_clients
        
        if args.split.mode == 'dirichlet':
            client_datasets = split_data_dirichlet(
                train_dataset, 
                num_clients,
                args.split.alpha
            )
        elif args.split.mode == 'iid':
            indices = list(range(len(train_dataset)))
            np.random.shuffle(indices)
            
            chunks = np.array_split(indices, num_clients)
            client_datasets = {
                i: Subset(train_dataset, chunk) 
                for i, chunk in enumerate(chunks)
            }
        else:
            print(f"Warning: Split mode '{args.split.mode}' not recognized. Using IID split.")
            indices = list(range(len(train_dataset)))
            np.random.shuffle(indices)
            chunks = np.array_split(indices, num_clients)
            client_datasets = {
                i: Subset(train_dataset, chunk) 
                for i, chunk in enumerate(chunks)
            }
        
        if hasattr(args.split, 'share_balanced_subset') and args.split.share_balanced_subset:
            samples_per_class = getattr(args.split, 'samples_per_class', 50)
            samples_per_client = getattr(args.split, 'samples_per_client', 20)
            
            balanced_indices = create_balanced_subset(train_dataset, samples_per_class=samples_per_class)
            
            client_datasets_with_split = {i: DatasetSplit(train_dataset, [idx for idx in client_datasets[i].indices]) 
                                         for i in range(num_clients)}
            
            client_datasets, shared_data = share_balanced_data(
                client_datasets_with_split, 
                balanced_indices, 
                samples_per_client=samples_per_client
            )
        
        print(f"Dataset split complete. Distribution:")
        for client_id, dataset in client_datasets.items():
            print(f"Client {client_id}: {len(dataset)} samples")
        
        datasets = {
            "train": client_datasets,
            "test": test_dataset,
        }
    else:
        datasets = {
            "train": {0: train_dataset},
            "test": test_dataset,
        }
    
    return datasets

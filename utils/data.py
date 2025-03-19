from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import random
from torch.utils.data import Dataset, Subset, DataLoader
import copy
from collections import defaultdict

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import torchvision.datasets.accimage as accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            if len(list(im.split())) == 1: im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]

class BaseDataset2(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []
        self.loader = loader

    def nb_classes(self):
        return len(set(self.ys))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = self.loader(self.im_paths[index])
        if self.transform is not None:
            im = self.transform(im)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]

class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, img):
        blur_factor = np.random.uniform(self.min, self.max)
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=blur_factor))

class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.class_dict = {}
        for idx in self.idxs:
            _, label = self.dataset[idx]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)
            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights

class DatasetSplitSubset(DatasetSplit):
    def __init__(self, dataset, idxs, subset_classes=None):
        self.dataset = dataset
        self.subset_classes = subset_classes
        self.class_dict = {}
        self.indices = []

        for idx in idxs:
            _, label = self.dataset[int(idx)]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)

            if subset_classes is not None and int(label) not in subset_classes:
                continue

            self.indices.append(idx)

            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.dataset[self.indices[item]]
        return image, label

class DatasetSplitMultiView(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)

class MultiViewDataset(Dataset):
    def __init__(self, dataset, transform=None, n_views=2):
        self.dataset = dataset
        self.transform = transform
        self.n_views = n_views

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform is not None:
            views = [self.transform(img) for _ in range(self.n_views)]
            return views, target
        return img, target

    def __len__(self):
        return len(self.dataset)

class ClassBalancedDataset(Dataset):
    def __init__(self, dataset, num_samples_per_class=None):
        self.dataset = dataset
        self.num_samples = len(dataset)
        
        self.class_indices = defaultdict(list)
        for idx in range(self.num_samples):
            _, label = dataset[idx]
            if torch.is_tensor(label):
                label = label.item()
            self.class_indices[label].append(idx)
        
        if num_samples_per_class is None:
            num_samples_per_class = min(len(indices) for indices in self.class_indices.values())
        
        self.indices = []
        for label, class_indices in self.class_indices.items():
            if len(class_indices) < num_samples_per_class:
                sampled_indices = np.random.choice(class_indices, num_samples_per_class, replace=True)
            else:
                sampled_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
            self.indices.extend(sampled_indices)
        
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def create_balanced_subset(dataset, samples_per_class=50):
    all_indices = list(range(len(dataset)))
    class_indices = defaultdict(list)
    
    for idx in all_indices:
        _, label = dataset[idx]
        if torch.is_tensor(label):
            label = label.item()
        class_indices[label].append(idx)
    
    balanced_indices = []
    for label, indices in class_indices.items():
        if len(indices) > samples_per_class:
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            selected = indices
        balanced_indices.extend(selected)
    
    return balanced_indices

def share_balanced_data(client_datasets, balanced_subset, samples_per_client=20):
    num_clients = len(client_datasets)
    shared_data = {}
    
    for client_id in range(num_clients):
        if len(balanced_subset) > samples_per_client:
            client_shared = np.random.choice(balanced_subset, samples_per_client, replace=False)
        else:
            client_shared = balanced_subset
        
        client_datasets[client_id] = torch.utils.data.ConcatDataset([
            client_datasets[client_id],
            DatasetSplit(client_datasets[client_id].dataset, client_shared)
        ])
        
        shared_data[client_id] = client_shared
    
    return client_datasets, shared_data

def get_strong_augmentation(size=32, s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return transform

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

def get_strong_augmentation_transforms(size=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform

def get_dataset(args, trainset, mode='iid'):
    set = args.dataset.name
    if 'leaf' not in set:
        directory = args.dataset.client_path + '/' + set + '/' + ('un' if args.split.unbalanced==True else '') + 'balanced'
        filepath = directory+'/' + mode + (str(args.split.class_per_client) if mode == 'skew' else '') + (str(args.split.alpha) if mode == 'dirichlet' else '') + (str(args.split.overlap_ratio) if mode == 'overlap' else '') + '_clients' +str(args.trainer.num_clients) +  (("_toyinform_" + str(args.split.toy_noniid_rate) + "_" + str(args.split.limit_total_classes)+ "_" +  str(args.split.limit_number_per_class)) if 'toy' in mode else "")   + '.txt'

        check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
        create_new_client_data = not check_already_exist or args.split.create_client_dataset

        if create_new_client_data == False:
            try:
                dataset = {}
                with open(filepath) as f:
                    for idx, line in enumerate(f):
                        dataset = eval(line)
            except:
                print("Have problem to read client data")
                create_new_client_data = True

        if create_new_client_data == True:
            if mode == 'iid':
                dataset = cifar_iid(trainset, args.trainer.num_clients)
            elif mode == 'overlap':
                dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
            elif mode == 'skew':
                class_per_client = args.split.class_per_client
                dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
            elif mode == 'dirichlet':
                if args.split.unbalanced==True:
                    dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
                else:
                    dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
            elif mode == 'toy_noniid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = True)
            elif mode == 'toy_iid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = False)                
            else:
                print("Invalid mode ==> please select in iid, skewNclass, dirichlet")
                return

            try:
                os.makedirs(directory, exist_ok=True)
                with open(filepath, 'w') as f:
                    print(dataset, file=f)
            except:
                print("Fail to write client data at " + directory)

        if hasattr(args.split, 'share_balanced_subset') and args.split.share_balanced_subset:
            balanced_indices = create_balanced_subset(trainset, samples_per_class=args.split.samples_per_class)
            client_datasets = {i: DatasetSplit(trainset, dataset[i]) for i in range(args.trainer.num_clients)}
            client_datasets, shared_data = share_balanced_data(
                client_datasets, 
                balanced_indices, 
                samples_per_client=args.split.samples_per_client
            )
            return client_datasets
        
        return dataset
    elif 'leaf' in set:
        return trainset.get_train_idxs()
    elif set == 'shakespeare':
        return trainset.get_client_dic()
def get_strong_augmentation(size=32, s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return transform

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

def get_strong_augmentation_transforms(size=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform

def get_dataset(args, trainset, mode='iid'):
    set = args.dataset.name
    if 'leaf' not in set:
        directory = args.dataset.client_path + '/' + set + '/' + ('un' if args.split.unbalanced==True else '') + 'balanced'
        filepath = directory+'/' + mode + (str(args.split.class_per_client) if mode == 'skew' else '') + (str(args.split.alpha) if mode == 'dirichlet' else '') + (str(args.split.overlap_ratio) if mode == 'overlap' else '') + '_clients' +str(args.trainer.num_clients) +  (("_toyinform_" + str(args.split.toy_noniid_rate) + "_" + str(args.split.limit_total_classes)+ "_" +  str(args.split.limit_number_per_class)) if 'toy' in mode else "")   + '.txt'

        check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
        create_new_client_data = not check_already_exist or args.split.create_client_dataset

        if create_new_client_data == False:
            try:
                dataset = {}
                with open(filepath) as f:
                    for idx, line in enumerate(f):
                        dataset = eval(line)
            except:
                print("Have problem to read client data")
                create_new_client_data = True

        if create_new_client_data == True:
            if mode == 'iid':
                dataset = cifar_iid(trainset, args.trainer.num_clients)
            elif mode == 'overlap':
                dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
            elif mode == 'skew':
                class_per_client = args.split.class_per_client
                dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
            elif mode == 'dirichlet':
                if args.split.unbalanced==True:
                    dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
                else:
                    dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
            elif mode == 'toy_noniid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = True)
            elif mode == 'toy_iid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = False)                
            else:
                print("Invalid mode ==> please select in iid, skewNclass, dirichlet")
                return

            try:
                os.makedirs(directory, exist_ok=True)
                with open(filepath, 'w') as f:
                    print(dataset, file=f)
            except:
                print("Fail to write client data at " + directory)

        if hasattr(args.split, 'share_balanced_subset') and args.split.share_balanced_subset:
            balanced_indices = create_balanced_subset(trainset, samples_per_class=args.split.samples_per_class)
            client_datasets = {i: DatasetSplit(trainset, dataset[i]) for i in range(args.trainer.num_clients)}
            client_datasets, shared_data = share_balanced_data(
                client_datasets, 
                balanced_indices, 
                samples_per_client=args.split.samples_per_client
            )
            return client_datasets
        
        return dataset
    elif 'leaf' in set:
        return trainset.get_train_idxs()
    elif set == 'shakespeare':
        return trainset.get_client_dic()



# class MultiViewDataInjector(object):
#     def __init__(self, *args):
#         self.transforms = args[0]
#         self.random_flip = transforms.RandomHorizontalFlip()

#     def __call__(self, sample, *with_consistent_flipping):
#         if with_consistent_flipping:
#             sample = self.random_flip(sample)
#         output = [transform(sample) for transform in self.transforms]
#         return output


# class GaussianBlur(object):
#     def __init__(self, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.k = kernel_size
#         self.r = radias

#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )

#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()


# def get_strong_augmentation_transforms(size=32):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
    
#     color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([color_jitter], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         normalize,
#     ])
    
#     return train_transform

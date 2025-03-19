# import torch
# from torchvision import datasets, transforms
# import os
# from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced,cifar_dirichlet_unbalanced, cifar_iid, cifar_overlap, cifar_toyset
# import torch.nn as nn
# import csv
# from typing import List, Dict
# import copy
# import json
# from collections import OrderedDict

# import numpy as np

# __all__ = ['DatasetSplit', 'DatasetSplitSubset', 'DatasetSplitMultiView', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice'
#                                                                                                             ]

# create_dataset_log = False

# class TransformTwice:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, inp):
#         out1 = self.transform(inp)
#         out2 = self.transform(inp)
#         return out1, out2


# class DatasetSplit(torch.utils.data.Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """

#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]
#         self.class_dict = {}
#         for idx in self.idxs:
#             _, label = self.dataset[idx]
#             if torch.is_tensor(label):
#                 label = str(label.item())
#             else:
#                 label = str(label)
#             if label in self.class_dict:
#                 self.class_dict[str(label)] += 1
#             else:
#                 self.class_dict[str(label)] = 1


#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label
    
#     @property
#     def num_classes(self):
#         return len(self.class_dict.keys())
    
#     @property
#     def class_ids(self):
#         return self.class_dict.keys()
    
#     def importance_weights(self, labels, pow=1):
#         class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
#         weights = (1/class_counts)**pow
#         weights /= weights.mean()
#         return weights



# class DatasetSplitSubset(DatasetSplit):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """

#     def __init__(self, dataset, idxs, subset_classes=None):
#         self.dataset = dataset

#         self.subset_classes = subset_classes

#         self.class_dict = {}
#         self.indices = []

#         for idx in idxs:
#             _, label = self.dataset[int(idx)]
#             if torch.is_tensor(label):
#                 label = str(label.item())
#             else:
#                 label = str(label)

#             if subset_classes is not None and int(label) not in subset_classes:
#                 continue

#             self.indices.append(idx)

#             if label in self.class_dict:
#                 self.class_dict[str(label)] += 1
#             else:
#                 self.class_dict[str(label)] = 1


#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.indices[item]]
#         return image, label
    
#     @property
#     def num_classes(self):
#         return len(self.class_dict.keys())
    
#     @property
#     def class_ids(self):
#         return self.class_dict.keys()
    
#     def importance_weights(self, labels, pow=1):
#         class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
#         weights = (1/class_counts)**pow
#         weights /= weights.mean()
#         return weights





# class DatasetSplitMultiView(torch.utils.data.Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """

#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         (view1, view2), label = self.dataset[self.idxs[item]]
#         return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)


# def get_dataset(args, trainset, mode='iid'):
#     set = args.dataset.name
#     if 'leaf' not in set:
#         directory = args.dataset.client_path + '/' + set + '/' + ('un' if args.split.unbalanced==True else '') + 'balanced'
#         filepath = directory+'/' + mode + (str(args.split.class_per_client) if mode == 'skew' else '') + (str(args.split.alpha) if mode == 'dirichlet' else '') + (str(args.split.overlap_ratio) if mode == 'overlap' else '') + '_clients' +str(args.trainer.num_clients) +  (("_toyinform_" + str(args.split.toy_noniid_rate) + "_" + str(args.split.limit_total_classes)+ "_" +  str(args.split.limit_number_per_class)) if 'toy' in mode else "")   + '.txt'


#         check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
#         create_new_client_data = not check_already_exist or args.split.create_client_dataset

#         if create_new_client_data == False:
#             try:
#                 dataset = {}
#                 with open(filepath) as f:
#                     for idx, line in enumerate(f):
#                         dataset = eval(line)
#             except:
#                 print("Have problem to read client data")

#         if create_new_client_data == True:

#             if mode == 'iid':
#                 dataset = cifar_iid(trainset, args.trainer.num_clients)
#             elif mode == 'overlap':
#                 dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
#             # elif mode[:4] == 'skew' and mode[-5:] == 'class':
#             elif mode == 'skew':
#                 class_per_client = args.split.class_per_client
#                 # assert class_per_client * args.trainer.num_clients == trainset.dataset.num_classes
#                 # class_per_user = int(mode[4:-5])
#                 dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
#             elif mode == 'dirichlet':
#                 if args.split.unbalanced==True:
#                     dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
#                 else:
#                     dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
#             elif mode == 'toy_noniid':
#                 dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = True)
#             elif mode == 'toy_iid':
#                 dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = False)                

            
#             else:
#                 print("Invalid mode ==> please select in iid, skewNclass, dirichlet")
#                 return

#             try:
#                 os.makedirs(directory, exist_ok=True)
#                 with open(filepath, 'w') as f:
#                     print(dataset, file=f)

#             except:
#                 print("Fail to write client data at " + directory)

#         return dataset
#     elif 'leaf' in set:
#         return trainset.get_train_idxs()
#     elif set == 'shakespeare':
#         return trainset.get_client_dic()



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
#     """blur a single image on CPU"""

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
import torch
from torchvision import datasets, transforms
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced, cifar_dirichlet_unbalanced, cifar_iid, cifar_overlap, cifar_toyset
import torch.nn as nn
import csv
from typing import List, Dict
import copy
import json
from collections import OrderedDict

import numpy as np

__all__ = ['DatasetSplit', 'DatasetSplitSubset', 'DatasetSplitMultiView', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice', 'create_balanced_subset', 'share_balanced_data']

create_dataset_log = False

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


class DatasetSplitMultiView(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)


def create_balanced_subset(dataset, samples_per_class=50):
    all_indices = list(range(len(dataset)))
    class_indices = {}
    
    for idx in all_indices:
        _, label = dataset[idx]
        if torch.is_tensor(label):
            label = label.item()
        
        if label not in class_indices:
            class_indices[label] = []
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


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output


class GaussianBlur(object):
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()


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

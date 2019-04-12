import torch
import torchvision.datasets 
import torchvision.transforms
import torch.utils.data

from PIL import Image 
import numpy as np

import pickle
import os
import sys

def extract_subclasses(subclasses, Y_coarse) : 
    indices = []
    
    for sc in subclasses :
        indices += [i for i in range(len(Y_coarse)) if Y_coarse[i] == sc]
    
    return indices

def create_subclass_dataset(dataset, data_loc, coarse_classes=[]) : 
    if dataset == 'cifar100' : 
        data_loc = os.path.join(data_loc, 'cifar-100-python')

        # training data
        with open(os.path.join(data_loc, 'train'), mode='rb') as train_data : 
            train_imgs = pickle.load(train_data, encoding='latin1')        
        
        train_X = train_imgs['data']
        train_fine_Y = train_imgs['fine_labels']
        train_coarse_Y = train_imgs['coarse_labels']
        train_filenames = train_imgs['filenames']
        
        # test data
        with open(os.path.join(data_loc, 'test'), mode='rb') as test_data : 
            test_imgs = pickle.load(test_data, encoding='latin1')        
        
        test_X = test_imgs['data']
        test_fine_Y = test_imgs['fine_labels']
        test_coarse_Y = test_imgs['coarse_labels']
        test_filenames = test_imgs['filenames']
        
        # meta data
        with open(os.path.join(data_loc, 'meta'), mode = 'rb') as meta_data : 
            meta_data = pickle.load(meta_data, encoding='latin1')
        
        fine_Y_names = meta_data['fine_label_names']
        coarse_Y_names = meta_data['coarse_label_names']

        # extract the images for those classes
        Y_labels = [coarse_Y_names.index(y) for y in coarse_classes] 
        train_indices = extract_subclasses(Y_labels, train_coarse_Y)
        test_indices = extract_subclasses(Y_labels, test_coarse_Y)

        return (train_indices, test_indices)

def import_and_preprocess_dataset(params) : 
    
    dataset = params.dataset 
    train_batch = params.train_batch 
    test_batch = params.test_batch 
    workers = params.workers
    data_loc = params.data_location
    
    assert dataset == 'cifar10' or dataset == 'cifar100'or dataset == 'imagenet', 'Dataset has to be one of cifar10, cifar100, imagenet'
    
    print('Preparing dataset %s' % dataset)

    # CIFAR10
    if dataset == 'cifar10' : 
        # data_loc = '/home/ar4414/multipres_training/organised/data'
        data_loader = torchvision.datasets.CIFAR10
        num_classes = 10 
        
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = data_loader(root=data_loc, train=True, download=False, transform=train_transform)
        test_set = data_loader(root=data_loc, train=False, download=False, transform=test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=workers)

    # CIFAR100
    elif dataset == 'cifar100' : 
        # data_loc = '/home/ar4414/multipres_training/organised/data'
        data_loader = torchvision.datasets.CIFAR100
        if params.sub_classes != [] : 
            print('Generating subset of dataset with classes %s' % params.sub_classes)
            train_indices, test_indices = create_subclass_dataset(dataset, data_loc, params.sub_classes) 
        num_classes = 100
        
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = data_loader(root=data_loc, train=True, download=False, transform=train_transform)
        test_set = data_loader(root=data_loc, train=False, download=False, transform=test_transform)

        if params.sub_classes != [] :     
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, num_workers=workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices))
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, num_workers=workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices))
        else : 
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=workers)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=workers)
    
    # ImageNet
    else : 
        # data_loc = '/mnt/storage/imagenet_original/data'
        if params.sub_classes != [] : 
            data_loc = create_subclass_dataset(dataset, data_loc, params.sub_classes) 
        # train_dir = os.path.join('/mnt/storage/imagenet_original/data', 'train')
        # test_dir = os.path.join('/mnt/storage/imagenet_original/data', 'validation')
        train_dir = os.path.join(data_loc, 'train')
        test_dir = os.path.join(data_loc, 'validation')
        num_classes = 1000
            
        train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
            
        test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        
        train_set = torchvision.datasets.ImageFolder(train_dir, train_transform)
        test_set = torchvision.datasets.ImageFolder(test_dir, test_transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=workers)

    return (train_loader, test_loader)

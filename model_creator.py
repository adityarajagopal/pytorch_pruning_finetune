import models 
import torch.nn
import torch.backends
import torchvision 

import pruning.methods as pruning

import sys
import os

def setup_model(params) : 
    if params.dataset == 'cifar10' : 
        import models.cifar as models 
        num_classes = 10

    elif params.dataset == 'cifar100' : 
        import models.cifar as models 
        num_classes = 100

    else : 
        import models.imagenet as models 
        num_classes = 1000

    print("Creating Model %s" % params.arch)
    
    if params.arch.endswith('resnet'):
        model = models.__dict__[params.arch](
                    num_classes=num_classes,
                    depth=params.depth
                )
    else:
        model = models.__dict__[params.arch](num_classes=num_classes)
    
    model = torch.nn.DataParallel(model, params.gpu_list)
    model = model.cuda()

    if params.resume == True or params.branch == True: 
        checkpoint = torch.load(params.pretrained)
        model.load_state_dict(checkpoint, strict=False)

    if params.evaluate == True: 
        checkpoint = torch.load(params.pretrained)
        model.load_state_dict(checkpoint['state_dict'])

    if params.finetune == True: 
        checkpoint = torch.load(params.pretrained)
        if 'state_dict' in checkpoint.keys() :
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else : 
            model.load_state_dict(checkpoint, strict=False)
    
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    optimiser = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    return (model, criterion, optimiser)
    




import sys
import time

import torch.autograd

import utils

def update_lr(params, optimiser) : 
    # update learning rate
    if params.lr_schedule != [] : 
        # get epochs to change at and lr at each of those changes
        # ::2 gets every other element starting at 0 
        change_epochs = params.lr_schedule[::2]
        new_lrs = params.lr_schedule[1::2]
        epoch = params.curr_epoch

        if epoch in change_epochs : 
            new_lr = new_lrs[change_epochs.index(epoch)]
            if new_lr == -1 :
                params.lr *= params.gamma
            else : 
                params.lr = new_lr
         
        for param_group in optimiser.param_groups : 
            param_group['lr'] = params.lr

    return params

def train(model, criterion, optimiser, inputs, targets) : 
    model.train()
    
    outputs = model(inputs) 
    loss = criterion(outputs, targets)

    prec1, prec5 = utils.accuracy(outputs.data, targets.data) 

    model.zero_grad() 
    loss.backward() 
    
    optimiser.step()

    return (loss, prec1, prec5)

def train_network(params, tbx_writer, checkpointer, train_loader, test_loader, model, criterion, optimiser) :  
    print('Epoch,\tLR,\tLoss,\tTop1,\tTop5')
    
    for epoch in range(params.start_epoch, params.epochs) : 
        # state['curr_epoch'] = epoch
        params.curr_epoch = epoch
        state = update_lr(params, optimiser)

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader) : 
            # move inputs and targets to GPU
            if params.use_cuda : 
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
            # train model
            loss, prec1, prec5 = train(model, criterion, optimiser, inputs, targets)
            
            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)

        params.loss = losses.avg        
        params.top1 = top1.avg        
        params.top5 = top5.avg        
        
        checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state())
        
        print('{},\t{},\t{},\t{},\t{}'.format(epoch, params.lr, losses.avg, top1.avg, top5.avg))






import sys
import time

import torch.autograd

import utils
import inference

import pruning.methods as pruning
import pruning.utils as pruning_utils

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

def train_network(params, checkpointer, train_loader, test_loader, model, criterion, optimiser) :  
    print('Config,\tEpoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5')
        
    for epoch in range(params.start_epoch, params.epochs) : 
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

        params.train_loss = losses.avg        
        params.train_top1 = top1.avg        
        params.train_top5 = top5.avg        

        if params.finetune == True : 
            # get test loss of subset on new model
            params.test_loss, params.test_top1, params.test_top5 = inference.test_network(params, test_loader['subset'], model, criterion, optimiser)
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state(), save_cp=True, config='11')
            print('{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format('11', epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5))

            # get test loss of entire dataset on new model
            params.test_loss, params.test_top1, params.test_top5 = inference.test_network(params, test_loader['orig'], model, criterion, optimiser)
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state(), save_cp=False, config='01')
            print('{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format('01', epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5))   
        else : 
            params.test_loss, params.test_top1, params.test_top5 = inference.test_network(params, test_loader, model, criterion, optimiser)
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state())
            print('{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format('00', epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5))


def finetune_network(params, checkpointer, train_loader, test_loader, model, criterion, optimiser) :  
    print('Config,\tEpoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5')
        
    for epoch in range(params.start_epoch, params.epochs) : 
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

        params.train_loss = losses.avg        
        params.train_top1 = top1.avg        
        params.train_top5 = top5.avg        

        if params.finetune == True : 
            if (params.prune_weights == True or params.prune_filters == True) and (epoch % 15) == 0 and epoch != 0 : 
            # if (params.prune_weights == True or params.prune_filters == True) and (epoch % 2) == 0 : 
                print('Pruning network')
                model = pruning.prune_model(params, model)
                params.pruning_perc += 10
                print('Pruned Percentage = {}'.format(pruning_utils.prune_rate(params, model)))

            # get test loss of subset on new model
            params.test_loss, params.test_top1, params.test_top5 = inference.test_network(params, test_loader['subset'], model, criterion, optimiser)
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state(), save_cp=True, config='11')
            print('{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format('11', epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5))
            params.tbx.add_scalar('__'.join(params.sub_classes)+'/top1_subset_on_new_model', params.test_top1, params.curr_epoch)
            
            # get test loss of entire dataset on new model
            params.test_loss, params.test_top1, params.test_top5 = inference.test_network(params, test_loader['orig'], model, criterion, optimiser)
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state(), save_cp=False, config='01')
            print('{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format('01', epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5))   
            params.tbx.add_scalar('__'.join(params.sub_classes)+'/top1_all_on_new_model', params.test_top1, params.curr_epoch)
        
        else : 
            params.test_loss, params.test_top1, params.test_top5 = inference.test_network(params, test_loader, model, criterion, optimiser)
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params.get_state())
            print('{},\t{},\t{},\t{},\t{},\t{},\t{},\t{},\t{}'.format('00', epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5))








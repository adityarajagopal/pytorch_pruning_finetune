import torch.autograd

import utils

def test_network(params, test_loader, model, criterion, optimiser) :  
    model.eval()
        
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(test_loader) : 
        # move inputs and targets to GPU
        if params.use_cuda : 
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # perform inference 
        outputs = model(inputs) 
        loss = criterion(outputs, targets)
        
        prec1, prec5 = utils.accuracy(outputs.data, targets.data)

        losses.update(loss) 
        top1.update(prec1) 
        top5.update(prec5)
        
    print('Loss: {}, Top1: {}, Top5: {}'.format(losses.avg, top1.avg, top5.avg))

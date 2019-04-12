import sys

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TeePrinting(object): 
    def __init__(self, logfile=None): 
        self.terminal = sys.stdout 
        self.logfile = logfile
        if self.logfile is not None : 
            self.log = open(self.logfile, 'a')

    def write(self, message): 
        if message[0] == '~' :
            if self.logfile is not None : 
                self.log.write(message[1:] + '\n')
            self.terminal.write(message[1:]) 
        else : 
            self.terminal.write(message) 

    def flush(self):
        pass

def accuracy(output, target) : 
    batch_size = target.size(0) 
    # torch.topk returns the values and indices of the k(5) largest elements in dimension 1 in a sorted manner
    _, indices = output.topk(5, 1, True, True)
    indices.t_()
    correct_predictions = indices.eq(target.view(1,-1).expand_as(indices))

    res = [] 
    for k in (1,5) : 
        correct_k = correct_predictions[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

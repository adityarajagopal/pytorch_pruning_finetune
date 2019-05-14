from __future__ import print_function

import argparse
import configparser as cp
import copy 

class Params() : 
    def __init__(self, config_file) : 
        # attributes read in from config file 
        self.dataset = config_file.get('dataset', 'dataset')
        self.data_location = config_file.get('dataset', 'dataset_location')

        self.arch = config_file.get('cnn', 'architecture')        
        self.depth = config_file.get('cnn', 'depth')       
        self.cardinality = config_file.get('cnn', 'cardinality')
        self.widen_factor = config_file.get('cnn', 'widen_factor')
        self.growth_rate = config_file.get('cnn', 'growth_rate')
        self.compression_rate = config_file.get('cnn', 'compression_rate')

        self.print_only = config_file.getboolean('training_hyperparameters', 'print_only')
        self.epochs = config_file.getint('training_hyperparameters', 'total_epochs')
        self.train_batch = config_file.getint('training_hyperparameters', 'train_batch')
        self.test_batch = config_file.getint('training_hyperparameters', 'test_batch') 
        self.lr = config_file.getfloat('training_hyperparameters', 'learning_rate')
        self.dropout = config_file.getfloat('training_hyperparameters', 'dropout_ratio')
        self.gamma = config_file.getfloat('training_hyperparameters', 'gamma')
        self.momentum = config_file.getfloat('training_hyperparameters', 'momentum') 
        self.weight_decay = config_file.getfloat('training_hyperparameters', 'weight_decay') 
        self.mo_schedule = [self.__to_num(i) for i in config_file.get('training_hyperparameters', 'momentum_schedule').split()]
        self.lr_schedule = [self.__to_num(i) for i in config_file.get('training_hyperparameters', 'lr_schedule').split()]
        
        self.sub_classes = config_file.get('pruning_hyperparameters', 'sub_classes').split() 
        self.this_layer_up = config_file.getint('pruning_hyperparameters', 'this_layer_up') 
        self.finetune = config_file.getboolean('pruning_hyperparameters', 'finetune')
        self.prune_weights = config_file.getboolean('pruning_hyperparameters', 'prune_weights')
        self.prune_filters = config_file.getboolean('pruning_hyperparameters', 'prune_filters')
        self.pruning_perc = config_file.getfloat('pruning_hyperparameters', 'pruning_perc')

        assert not (self.prune_weights == True and self.prune_filters == True), 'Cannot prune both weights and filters'

        self.tbx_name = config_file.get('pytorch_parameters', 'tbx_name')
        self.enable_tbx = config_file.getboolean('pytorch_parameters', 'enable_tbx')
        self.manual_seed = config_file.getint('pytorch_parameters', 'manual_seed')
        self.workers = config_file.getint('pytorch_parameters', 'data_loading_workers')
        self.gpu_id = config_file.get('pytorch_parameters', 'gpu_id')
        self.pretrained = config_file.get('pytorch_parameters', 'pretrained')        
        self.checkpoint = config_file.get('pytorch_parameters', 'checkpoint_path')
        self.test_name = config_file.get('pytorch_parameters', 'test_name')
        self.resume = config_file.getboolean('pytorch_parameters', 'resume')
        self.branch = config_file.getboolean('pytorch_parameters', 'branch')
        self.evaluate = config_file.getboolean('pytorch_parameters', 'evaluate')
        self.tee_printing = config_file.get('pytorch_parameters', 'tee_printing')

        # attributes used internally
        self.use_cuda = True
        self.gpu_list = []
        self.device = 'cuda:0'
        self.tbx = None
        self.pruned_filters = {}
        self.plots = {}
        self.prune_rate_by_layer = []
        
        self.start_epoch = 0 
        self.curr_epoch = 0 
        self.train_loss = 0 
        self.train_top1 = 1
        self.train_top5 = 1
        self.test_loss = 0 
        self.test_top1 = 1
        self.test_top5 = 1

    def get_state(self) : 
        state = {}
        for key, val in self.__dict__.items():
            if key != 'tbx':
                state[key] = val
        return state

    def __to_num(self, x) : 
        try : 
            return int(x) 
        except ValueError: 
            return float(x) 

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')

    # Command line vs Config File
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    
    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    
    # Optimization options
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize')
    
    # Hyperparameter tuning
    parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--mo-schedule', type=float, nargs='+', default=None, help='Fixed values for lr at corresponding schedule epoch.')
    parser.add_argument('--mo-list', type=float, nargs='+', default=None, help='Fixed values for lr at corresponding schedule epoch.')
    
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    
    # Architecture
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                         ' | '.join(model_names) +
    #                         ' (default: resnet18)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--sf', type=float, default=1.0, metavar='N')
    parser.add_argument('--c16', type=int, default=350, metavar='N', help='number of iterations of fp16 to perform')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    
    #Device options
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    
    # Frequently used arguments   
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--max-lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--delta', default=0.04, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=None, help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr-schedule', type=float, nargs='+', default=None, help='Fixed values for lr at corresponding schedule epoch.')
    parser.add_argument('--prec-schedule', type=float, nargs='+', default=None, help='Epochs at which precision change should happen')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--bw', type=int, default=4, metavar='N', help='quantisation bit width')
    parser.add_argument('--quant', type=int, default=1, metavar='N', help='perform quantisation or not')
    parser.add_argument('--resolution', type=int, default=3, metavar='N', help='epochs over which gd is calculated')
    parser.add_argument('--lr-resolution', type=int, default=5, metavar='N', help='epochs over which gd is calculated')
    parser.add_argument('--prec-thresh', type=float, default=1.3, metavar='N', help='drop in gd that causes precision change')
    parser.add_argument('--lr-var-thresh', type=float, default=0.03, metavar='N', help='drop in gd that causes precision change')
    parser.add_argument('--prec-count-thresh', type=int, default=20, metavar='N', help='number of epochs after which force change precision')
    parser.add_argument('--low-prec-limit', type=int, default=8, metavar='N', help='number of epochs after which force change precision')
    parser.add_argument('--patience', type=int, default=2, metavar='N', help='number of epochs after which force change precision')
    parser.add_argument('-e', '--evaluate', type=str, default=None, metavar='PATH', help='evaluate model on validation set')
    
    args = parser.parse_args()

    return args

def parse_config_file(config_file) : 
    config = cp.ConfigParser() 
    config.read(config_file)
    return Params(config)





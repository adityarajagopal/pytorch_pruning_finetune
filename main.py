import os
import random
import sys

import param_parser as pp
import input_preprocessor as preproc
import model_creator as mc
import training
import inference
import utils
from checkpointing import Checkpointer
import pruning.utils as pruning_utils

import tensorboardX as tbx

import torch
import torch.cuda

def main() : 
    # parse config
    print('==> Parsing Config File')
    args = pp.parse_command_line_args()
    if args.config_file != 'None' : 
        params = pp.parse_config_file(args.config_file)
    else : 
        raise ValueError('Need to specify config file with parameters')
    
    # TODO: need to make cli consistent with final version of config file else : 
        # params = args 
        # state = {k: v for k,v in args._get_kwargs()}

    # call checkpointer to see if state is from parsed parameters or checkpoint  
    checkpointer = Checkpointer(params)
    params = checkpointer.restore_state(params)

    # setup hardware 
    use_cuda = torch.cuda.is_available()
    params.use_cuda = use_cuda
    if use_cuda : 
        print('==> Using GPU %s' % params.gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id
        params.gpu_list = [int(x) for x in params.gpu_id.split(',')]
        if len(params.gpu_list) == 1:
            torch.cuda.set_device(params.gpu_list[0])
    else : 
        print('==> No CUDA GPUs found --> Using CPU')

    # setup random number generator
    print('==> Setting up random number seed')
    if params.manual_seed is None or params.manual_seed < 0 : 
        params.manual_seed = random.randint(1, 10000)
    random.seed(params.manual_seed) 
    torch.manual_seed(params.manual_seed)
    if use_cuda : 
        torch.cuda.manual_seed_all(params.manual_seed)

    # setup model 
    print('==> Setting up Model')
    model, criterion, optimiser = mc.setup_model(params)
    
    # preprocess dataset
    print('==> Setting up Input Dataset')
    train_loader, test_loader = preproc.import_and_preprocess_dataset(params) 

    # setup tee printing so some prints can be written to a log file
    if (params.tee_printing != 'None') : 
        print('==> Tee Printing Enabled to logfile {}'.format(params.tee_printing))
        sys.stdout = utils.TeePrinting(params.tee_printing)

    # setup tensorboardX   
    params.tbx = tbx.SummaryWriter(comment='-prune_weights_2+_int_15_epochs_60_base_prune_20_' + '__'.join(params.sub_classes))

    if params.finetune == True : 
        # every 10 epochs, prune and then retrain 
        print('==> Performing Finetuning')
        training.finetune_network(params, checkpointer, train_loader, test_loader, model, criterion, optimiser) 
        params.tbx.add_custom_scalars_multilinechart(params.prune_rate_by_layer, category='__'.join(params.sub_classes), title='prune rate for different layers across epochs')
        
    elif params.evaluate == False : 
        # train model 
        print('==> Performing Training')
        training.train_network(params, checkpointer, train_loader, test_loader, model, criterion, optimiser) 
    else : 
        # perform inference only
        print('==> Performing Inference')
        inference.test_network(params, test_loader, model, criterion, optimiser)

    params.tbx.close()
        
main()
         

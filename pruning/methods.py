import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import matplotlib.pyplot as plt

from pruning.utils import prune_rate, arg_nonzero_min

def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    this_layer_up = 2
    layer_count = 0
    for p in model.parameters():
        if len(p.data.size()) != 1:
            if layer_count >= this_layer_up :
                pruned_inds = p.data.abs() > threshold
                mask = pruned_inds.float()
            else :
                mask = torch.tensor((), dtype=torch.float32)
                mask = mask.new_ones(p.size())

            layer_count += 1            
            
            masks.append(mask)
    
    return masks

def prune_one_filter(params, model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():
        if len(p.data.size()) == 4 : # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                tmp = torch.tensor((), dtype=torch.float32)
                # masks.append(np.ones(p_np.shape).astype('float32'))
                masks.append(tmp.new_ones(p_np.shape))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            
            values.append([min_value, min_ind])
        
    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    # to_prune_layer_ind = np.argmin(values[:, 0])
    this_layer_up = 2
    to_prune_layer_ind = np.argmin(values[this_layer_up:, 0])
    to_prune_layer_ind += this_layer_up
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    # params.pruned_layers.append(to_prune_layer_ind)

    # print('Prune filter #{} in layer #{}'.format(
    #     to_prune_filter_ind, 
    #     to_prune_layer_ind))

    return masks


def filter_prune(params, model):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.
    params.pruned_layers = []

    while current_pruning_perc < params.pruning_perc:
        masks = prune_one_filter(params, model, masks)
        model.module.set_masks(masks)
        current_pruning_perc = prune_rate(params, model, verbose=False)
    
    return masks

def prune_model(params, model) : 
    if params.prune_weights == True : 
        print('Creating Weight Pruning Mask')
        masks = weight_prune(model, params.pruning_perc)
        model.module.set_masks(masks)
    elif params.prune_filters == True : 
        print('Creating Filter Pruning Mask')
        masks = filter_prune(params, model)
        # pruned_layers = torch.tensor(params.pruned_layers, dtype=torch.int32)
        # params.tbx.add_histogram(str(params.sub_classes)+'/pruned_layers', pruned_layers, params.curr_epoch, bins='auto')

    return model

def plot_weight_mask(model, layer_count, mask):
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'linear']
    layer = layer_names[layer_count]
    new_mask = mask.byte()

    new_mask_np = new_mask.cpu().numpy() 
    for filters in range(new_mask_np.shape[0]) : 
        for channels in range(new_mask_np.shape[1]) : 
            plt.imshow(new_mask_np[filters][channels], cmap='hot', interpolation='nearest')
            img = str(filters) + '_' + str(channels)
            plt.savefig('img/' + img + '.png')
    sys.exit()
     
    return new_mask

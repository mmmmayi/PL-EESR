# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import importlib

import torch

from asvtorch.src.settings.settings import Settings
import asvtorch.src.misc.fileutils as fileutils

def load_network_multiGPU(epoch: int,device,n_speakers,feat_dim):
    model_filepath = os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))
    loaded_states = torch.load(model_filepath, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = loaded_states['model_state_dict']
    for k, v in state_dict.items():
        namekey = k[7:]
        #print('debug for name:',k)
        new_state_dict[namekey]=v
    net = initialize_net(feat_dim, n_speakers)
    net.to(device)
    net.load_state_dict(new_state_dict)
    return net



def load_network(epoch: int, device, n_speakers):
    model_filepath = os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))
    print('debug for model_filepath:',model_filepath)
    #model_filepath = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/pretrained/stage2.pt'
    loaded_states = torch.load(model_filepath, map_location=device)
    feat_dim = 30
    state_dict = loaded_states['model_state_dict']
    net = initialize_net(feat_dim, n_speakers)
    net.to(device)
    net.load_state_dict(state_dict)
    return net

def save_state(filename, epoch, net, optimizer1 ):
    model_dict = {'model_state_dict': net.state_dict(), 'optimizer1_state_dict': optimizer1.state_dict()}
    filename = fileutils.ensure_ext('{}.{}'.format(fileutils.remove_ext(filename, '.pt'), epoch), '.pt')
    torch.save(model_dict, filename)
    print('x-vector extractor model saved to: {}'.format(filename))

def load_state(filename, epoch, net, optimizer1,  device):
    filename = fileutils.ensure_ext('{}.{}'.format(fileutils.remove_ext(filename, '.pt'), epoch), '.pt')
    loaded_states = torch.load(filename)
    net.load_state_dict(loaded_states['model_state_dict'])
    optimizer1.load_state_dict(loaded_states['optimizer1_state_dict'])
   

# This allows to select the network class by using the class name in Settings
def initialize_net(feat_dim: int, n_speakers: int):
    module, class_name = Settings().network.network_class.rsplit('.', 1)
    print('initialize_net:{},{}'.format(module,class_name))
    FooBar = getattr(importlib.import_module(module), class_name)
    return FooBar(feat_dim, n_speakers)

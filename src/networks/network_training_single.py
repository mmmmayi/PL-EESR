# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import time
import random
import sys
import builtins
from tqdm import tqdm
import numpy as np
import torch
#torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from asvtorch.src.networks.training_dataloader_single import get_dataloader as get_training_dataloader
from asvtorch.src.networks.network_testing import validation_dataloader, compute_losses_and_accuracies
import asvtorch.src.networks.network_io as network_io
from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.mask.utils import extract
#from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
import asvtorch.src.misc.fileutils as fileutils

#def train_network(training_data: UtteranceList, writer, resume_epoch: int = 0):
def train_network(training_data: UtteranceList, mean, std, mean_lps, std_lps, writer, resume_epoch: int = 0):
    mean = mean.to(Settings().computing.device)
    std = std.to(Settings().computing.device)
    mean_lps = mean_lps.to(Settings().computing.device)
    std_lps = std_lps.to(Settings().computing.device)

    settings = Settings().network

#    training_data.convert_labels_to_numeric()

    n_speakers = training_data.get_number_of_speakers()
    #print('Number of speakers: {}'.format(n_speakers))

    #feat_dim = FeatureLoader().get_feature_dimension(training_data[0])
    feat_dim = 201
    # Training & validation:
    print('length of training_data',len(training_data))
    training_data, validation_data = _split_to_train_and_validation(training_data, settings.validation_utterances)

    # Subset of training:
    training_data_subset = _select_random_subset(training_data, settings.validation_utterances)

    #print('test for data subset: length of validation_data:{},length of training_data_subset:{}'.format(len(validation_data),len(training_data_subset)))
    
    training_dataloader = get_training_dataloader(training_data,Settings().network.minibatch_size)
    '''
    for label,mfcc in training_dataloader:
        print('in training_data, label shape:{}, label:{}'.format(label.size(),label))
        print('in training_data, mfcc shape:{}'.format(mfcc.size()))
        break
    validation_dataloader_1 = get_validation_dataloader(training_data_subset)
    for label,mfcc in validation_dataloader_1:
        print('in valdation_data, label shape:{}, label:{}'.format(label.size(),label))
        print('in validation_data, mfcc shape:{}'.format(mfcc.size()))
        break
    return
    '''
#    validation_dataloader_1 = get_training_dataloader(training_data_subset, 1)
#    validation_dataloader_2 = get_training_dataloader(validation_data, 1)

    validation_dataloader_1 = validation_dataloader(training_data_subset, int(Settings().network.minibatch_size/4))
    validation_dataloader_2 = validation_dataloader(validation_data, int(Settings().network.minibatch_size/4))
#    torch.multiprocessing.set_start_method('fork',force=True)
    print('prepare dataloader finished')
    net = network_io.initialize_net(feat_dim, n_speakers)
#    filename = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/vox2_clean/networks/epoch.44.pt'
#    pretrained = torch.load(filename) 
#    net.load_state_dict(pretrained['model_state_dict'])
    net.to(Settings().computing.device)
    #print('debug for fix model')
    print_learnable_parameters(net)
    #print(net)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of trainable parameters: {}'.format(total_params))

    criterion1 =  nn.CrossEntropyLoss()
#    criterion1 = nn.MSELoss(reduction='sum')
    criterion2 = nn.MSELoss(reduction='mean')
    criterion3 = nn.MSELoss(reduction='sum')

    optimizer1 = _init_optimizer(net, settings, settings.initial_learning_rate1)
#    scheduler = MultiStepLR(optimizer=optimizer2, milestones=[2, 4, 6, 8], gamma=0.1)
    log_folder = fileutils.get_network_log_folder()
    network_folder = fileutils.get_network_folder()
    output_filename = os.path.join(network_folder, 'epoch')

    if resume_epoch < 0:
        resume_epoch = -resume_epoch
        print('Computing ASV metrics for epoch {}...'.format(resume_epoch))
        network_io.load_state(output_filename, resume_epoch, net, optimizer1,  Settings().computing.device)
        return net, False, resume_epoch
    elif resume_epoch > 0:
        print('Resuming network training from epoch {}...'.format(resume_epoch))
        network_io.load_state(output_filename, resume_epoch, net, optimizer1,  Settings().computing.device)

    #net = nn.DataParallel(net, device_ids=Settings().computing.gpu_ids)
    
    net.train()
    for epoch in range(1, settings.epochs_per_train_call + 1):

        start_time = time.time()
        #print('Setting initial learning rates for this epoch...')
        current_learning_rate1 = optimizer1.param_groups[0]['lr']
        # start_lr, end_lr = _get_learning_rates_for_epoch(epoch + resume_epoch, settings)
        # current_learning_rate = start_lr
        # _update_learning_rate(optimizer, current_learning_rate, settings)

        logfilename = os.path.join(log_folder, 'epoch.{}.log'.format(epoch + resume_epoch))
        logfile = open(logfilename, 'w')
        print('Log file created: {}'.format(logfilename))

        print('Shuffling training data...')
        training_dataloader.dataset.shuffle(epoch+resume_epoch)

        
        optimizer1.zero_grad()
        training_loss = 0
        # For automatic learning rate scheduling:
        losses = []
        losses1,losses2,losses3,losses4,losses5 = 0,0,0,0,0
        lam = settings.lam
 
        print('Iterating over training minibatches...')
        print('loss = {}*(loss1)+{}*(loss3+loss4)'.format(str(lam),str(1-lam))) 
#        print('loss = loss3')
        print('length of dataloader:{}'.format(len(training_dataloader)))
        for i, batch in enumerate(training_dataloader):
            spk = batch['spk']
            ids = batch['id']
            n_frame = batch['len']
            features = batch['features'].to(Settings().computing.device)
            lps = batch['clean_lps'].to(Settings().computing.device)
            features = features.float()
            lps = lps.float()

                  
            features = (features-mean)/std
            lps = (lps-mean_lps)/std_lps
            speaker_labels = spk.to(Settings().computing.device)
            
            
      
            #print('feature shape:{}'.format(features.shape))
            '''
            o_lps = net(features, lps, mean_lps.to(Settings().computing.device), std_lps.to(Settings().computing.device), 'enhance')
            loss2 = compute_highEnergy(o_lps,lps)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            '''
            noisy_spk, clean_spk, noisy_hidden, clean_hidden, target_hidden = net(features, lps, mean_lps, std_lps)
            #noisy_spk, clean_spk, noisy_hidden, clean_hidden, target_hidden = net(features, lps)

            #loss1 = criterion3(noisy_embedding, clean_embedding)
            #loss2 = criterion3(clean_embedding, target_embedding)
            loss1 = torch.zeros(1).to(Settings().computing.device)
            loss2 = torch.zeros(1).to(Settings().computing.device)
            for hidden_num in range(len(target_hidden)):
                loss1 = loss1+criterion2(noisy_hidden[hidden_num], target_hidden[hidden_num])
                loss2 = loss2+criterion2(clean_hidden[hidden_num], target_hidden[hidden_num])
            loss3 = 3*criterion1(noisy_spk, speaker_labels)
            loss4 = 3*criterion1(clean_spk, speaker_labels)
            loss = loss1
            
            #print('loss1:{},loss3:{},loss4:{}'.format(loss1.item(),loss3.item(),loss4.item()))
            '''
            if torch.isnan(loss):
                print(ids)
                #print(features)
                import matplotlib.pyplot as plt
                import librosa.display
                for j in range(features.shape[0]):
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    librosa.display.specshow(features[j].squeeze().cpu().numpy(), y_axis='log', x_axis='time',sr=16000, ax=ax)                    
                    plt.savefig("check/"+ids[j]+'.png')
            '''
            #print('loss1:{},loss2:{},loss3:{},loss4:{}'.format(str(loss1.item()),str(loss2.item()),str(loss3.item()),str(loss4.item())))
            
            loss.backward()
            #print(net.mask_model.fconn.bias.grad)
#            torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            optimizer1.step()
            optimizer1.zero_grad()
            losses1 += loss1.item()
            #losses2 += loss2.item()
           
            losses3 += loss3.item()
            losses4 += loss4.item()

            losses.append(loss.item())
#            training_loss += (minibatch_loss1+minibatch_loss2)/2
            training_loss +=loss.item()
            #print('{} debug training_loss: {}'.format(i,training_loss))

            # Computing train and test accuracies and printing status:
            if i % settings.print_interval == settings.print_interval - 1:
                writer.add_scalar('mse_noisy_clean', losses1/(settings.print_interval*Settings().network.minibatch_size), i+(epoch + resume_epoch-1)*len(training_dataloader))
                #writer.add_scalar('mse_clean_target', losses2/(settings.print_interval*Settings().network.minibatch_size), i+(epoch + resume_epoch-1)*len(training_dataloader))
                writer.add_scalar('ce_noisy', losses3/(settings.print_interval*Settings().network.minibatch_size), i+(epoch + resume_epoch-1)*len(training_dataloader))
                writer.add_scalar('ce_clean', losses4/(settings.print_interval*Settings().network.minibatch_size), i+(epoch + resume_epoch-1)*len(training_dataloader))
                losses1,losses2,losses3,losses4 = 0,0,0,0
#                writer.add_scalar('loss', training_loss/settings.print_interval, i+(epoch + resume_epoch-1)*len(training_dataloader))

                if i % (settings.print_interval * settings.accuracy_print_interval) == settings.print_interval * settings.accuracy_print_interval - 1:
                    torch.cuda.empty_cache()
                    val_data = compute_losses_and_accuracies((validation_dataloader_1,validation_dataloader_2), net, criterion3, criterion1, mean, std, mean_lps, std_lps)
                    #val_data = compute_losses_and_accuracies((validation_dataloader_1,validation_dataloader_2), net, criterion3, criterion1)
                    output = 'Epoch {}, Time: {:.0f} s, Batch {}/{}, lr1: {:.6f},  train-loss: {:.3f}, subset-noisy-mse: {:.3f}, subset-clean-mse: {:.3f}, subset-noisy-ce: {:.3f}, subset-clean-ce: {:.3f}, subset-noisy-acc: {:.3f}, subset-clean-acc: {:.3f}, val-noist-mse: {:.3f}, val-clean-mse: {:.3f}, val-noisy-ce: {:.3f}, val-clean-ce: {:.3f}, val-noisy-acc: {:.3f}, val-clean-acc: {:.3f},'.format(epoch + resume_epoch, time.time() - start_time, i + 1, len(training_dataloader), current_learning_rate1, training_loss / settings.print_interval, val_data[0][0], val_data[0][1], val_data[0][2], val_data[0][3], val_data[0][4], val_data[0][5], val_data[1][0],val_data[1][1], val_data[1][2], val_data[1][3], val_data[1][4], val_data[1][5])
                    torch.cuda.empty_cache()
                else:
                    output = 'Epoch {}, Time: {:.0f} s, Batch {}/{}, lr1: {:.6f}, train-loss: {:.3f}'.format(epoch + resume_epoch, time.time() - start_time, i + 1, len(training_dataloader), current_learning_rate1, training_loss / (settings.print_interval*Settings().network.minibatch_size))
                training_loss = 0        
                print(output)
                logfile.write(output + '\n')
                
#        scheduler.step() 
        # Learning rate update:
        #print(net.mask_model.fconn.bias)
        prev_loss1 = net.training_loss.item()
        current_loss1 = np.asarray(losses).mean()
        room_for_improvement = max(Settings().network.min_room_for_improvement, prev_loss1 - Settings().network.target_loss)
        loss_change1 = (prev_loss1 - current_loss1) / room_for_improvement
        print('prev_loss1:{}, current_loss1:{}'.format(str(prev_loss1),str(current_loss1)))
        print('Average training loss reduced {:.2f}% from the previous epoch.'.format(loss_change1*100))
#        for param_group in optimizer2.param_groups:
#            param_group['lr'] = param_group['lr']/2
        if loss_change1 < Settings().network.min_loss_change_ratio:
            for param_group in optimizer1.param_groups:
                param_group['lr'] = param_group['lr']/2
            net.consecutive_lr_updates[0] += 1
#            print('Because loss change {:.2f}% <= {:.2f}%, the learning rate is halved: {} --> {}'.format(loss_change*100, Settings().network.min_loss_change_ratio*100, optimizer.param_groups[0]['lr'] * 2, optimizer.param_groups[0]['lr']))
            print('Consecutive LR updates: {}'.format(net.consecutive_lr_updates))
        else:
            net.consecutive_lr_updates[0] = 0
#        if loss_change2 < Settings().network.min_loss_change_ratio:
#            for param_group in optimizer2.param_groups:
#                param_group['lr'] = param_group['lr']/2

        net.training_loss[0] = current_loss1
        print(current_loss1)
        network_io.save_state(output_filename, epoch + resume_epoch, net, optimizer1)

#        if net.consecutive_lr_updates >= Settings().network.max_consecutive_lr_updates:
            #print('Stopping training because loss did not improve more than {:.3f}% ...'.format(Settings().network.min_loss_change * 100))
#            print('Stopping training because reached {} consecutive LR updates!'.format(Settings().network.max_consecutive_lr_updates))
#            return net, True, epoch + resume_epoch
   
    logfile.close()
    return net, False, epoch + resume_epoch


# def _get_learning_rates_for_epoch(epoch, settings):
#     if epoch <= len(settings.learning_rate_schedule):
#         start_lr = settings.learning_rate_schedule[epoch-1]
#     else:
#         start_lr = settings.learning_rate_schedule[-1]
#     if epoch + 1 <= len(settings.learning_rate_schedule):
#         end_lr = settings.learning_rate_schedule[epoch]
#     else:
#         end_lr = settings.learning_rate_schedule[-1]
#     return start_lr, end_lr

def compute_highEnergy(output,label):
    batch, frequency, feat = output.shape
    max_energy = torch.amax(label,dim=(1,2))+torch.log(torch.tensor(0.01))
    max_energy = max_energy.reshape(batch,1,1)
    index = label[:].ge(max_energy[:])
    result_output = output
    result_label = label
    num = torch.sum(index == True)
    loss = torch.sum(torch.pow((result_output[index]-result_label[index]),2))/num
    return loss

def plot_highEnergy(lps):
    import matplotlib.pyplot as plt
    import librosa.display
    shape = lps.shape
    lps = lps.cpu()
    max_energy = torch.amax(lps,dim=(1,2))+torch.log(torch.tensor(0.01))
    temp=torch.full([shape[1],shape[2]],0.)
    for i in range(shape[0]):
        clean=lps[i,:,:]
        clean_high=torch.where(clean>max_energy[i],clean,temp)
        print(clean)
        print(clean_high)
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        librosa.display.specshow(clean.numpy().T, y_axis='log', sr=16000, x_axis='time', ax=ax[0])
        ax[0].set(title='whole')
        librosa.display.specshow(clean_high.numpy().T, y_axis='log', sr=16000, x_axis='time', ax=ax[1])
        ax[1].set(title='high energy') 
        plt.savefig('/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/check/'+str(i)+'.png')

def _shuffle_data(segment_ids, speaker_labels, seed):
    random.seed(seed)
    c = list(zip(segment_ids, speaker_labels))
    random.shuffle(c)
    return zip(*c)

def _split_to_train_and_validation(data, n_validation, seed=101):
    np.random.seed(seed)
    n_utterances = len(data.utterances)
    print('length:',n_utterances)
    #validation_indices = np.random.choice(n_utterances, n_validation, replace=True)
    validation_indices = np.random.choice(n_utterances, n_validation, replace=False)
    training_indices = np.setdiff1d(np.arange(n_utterances), validation_indices)
    n_validation = validation_indices.size
    n_training = training_indices.size
    print('Training set of {} utterances divided randomly to sets of {} and {} for training and validation.'.format(n_utterances, n_training, n_validation))
    training_data = UtteranceList([data[i] for i in np.nditer(training_indices)], name='voxceleb1_mask_aug')
    validation_data = UtteranceList([data[i] for i in np.nditer(validation_indices)], name='voxceleb1_mask_aug')
    return training_data, validation_data

def _select_random_subset(data, n):
    print('Selecting random subset of training data for accuracy computation...')
    indices = np.random.choice(len(data.utterances), n, replace=False)
    subset_data = UtteranceList([data[i] for i in np.nditer(indices)], name='voxceleb1_mask_aug')
    return subset_data

# def _update_learning_rate(optimizer, learning_rate, settings):
#     optimizer.param_groups[0]['lr'] = learning_rate * settings.learning_rate_factor_for_frame_layers * settings.general_learning_rate_factor
#     optimizer.param_groups[1]['lr'] = learning_rate * settings.learning_rate_factor_for_pooling_layer * settings.general_learning_rate_factor
#     optimizer.param_groups[2]['lr'] = learning_rate * settings.general_learning_rate_factor

def _init_optimizer(net, settings, learning_rate):
    params = get_weight_decay_param_groups(net, settings.weight_decay_skiplist)
    if settings.optimizer == 'sgd':
        return optim.SGD(params, lr=learning_rate, weight_decay=settings.weight_decay, momentum=settings.momentum)
    if settings.optimizer == 'Adadelta':

        return optim.Adadelta(net.parameters(), lr=learning_rate)
        
    if settings.optimizer == 'Adam':
        return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)
    sys.exit('Unsupported optimizer: {}'.format(settings.optimizer))

def get_weight_decay_param_groups(model, skip_list):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if builtins.any(x in name for x in skip_list):
            no_decay.append(param)
            print('No weight decay applied to {}'.format(name))
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': Settings().network.weight_decay}]

def print_learnable_parameters(model: torch.nn.Module):
    print('Learnable parameters of the model:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #bandwidth /= kernel_mul ** (kernel_num // 2)
    #bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    band_para = [i-int(kernel_num/2) for i in range(kernel_num)]
    bandwidth_list = [bandwidth* (10**i) for i in band_para]
    # 高斯核的公式，exp(-|x-y|^2/2*bandwith)
    kernel_val = [torch.exp(-L2_distance / 2*bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 

def mmd(source, target, kernel_num=5,kernel_mul=2.0, fix_sigma=1):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, 	
                             	kernel_num=kernel_num, 	
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size] # Source<->Source
    YY = kernels[batch_size:, batch_size:] # Target<->Target
    XY = kernels[:batch_size, batch_size:] # Source<->Target
    YX = kernels[batch_size:, :batch_size] # Target<->Source
    loss = torch.mean(XX + YY - XY -YX) 
    return loss

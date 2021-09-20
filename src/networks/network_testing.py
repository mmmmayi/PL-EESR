# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).
import math
import os
from typing import Tuple, Type, Union
import time
import kaldiio
import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import librosa
from asvtorch.src.misc.miscutils import test_finiteness
from asvtorch.src.mask.utils import extract
from asvtorch.src.utterances.utterance_list import UtteranceList
#from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
from asvtorch.src.settings.settings import Settings
from asvtorch.src.ivector.sufficient_stats import SufficientStats

class ValidationDataset(Dataset):
    def __init__(self, data: UtteranceList,batch_size):	
        super().__init__()
        data.sort(reverse=False)
        self.data = data
        self.path_list = ['/data_a8/mayi/mix_speech_voxceleb/voxceleb1_fea','/data_a8/mayi/mix_speech_voxceleb/voxceleb2_fea']
        self.batch_data = []
        self.name = data.name
        start = 0
        #number_of_minibathes = math.ceil(Settings().network.utts_per_speaker_in_epoch * len(self.data) / Settings().network.minibatch_size)
        number_of_minibathes = math.ceil(len(self.data) / 1)
        index = -1
        num = 0
        for _ in range(number_of_minibathes):
            self.batch_data.append([])
            for _ in range(batch_size):
                if index==len(self.data)-1:
                    index = index-64
                    num += 1
                else:
                    index = index+1
                #print('length of utt_temp:{},index:{}'.format(len(self.data),index))

                self.batch_data[-1].append(self.data.utterances[index])
        #print('debug mask dataset, length of train dataset:{}, length of train dataset[0]:{}'.format(len(self.batch_data),len(self.batch_data[0])))
        print('broken time:{}'.format(num))

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, index):
        utt = self.batch_data[index][0]
        spk = utt.spk_id
        sample = {
            "id": utt.utt_id,
            "spk": spk,
            "len":utt.n_signal
        }
        return sample

class TestDataset(Dataset):
    def __init__(self, data: UtteranceList, length, path_list, usage='validation'):
        assert usage in ['extraction', 'validation', 'sitw']
        self.usage = usage
        self.batch_data = []
        self.path_list = path_list
        self.name = data.name
        
        data.sort(reverse=False) # Sorting based on utterance length
        if usage == 'extraction' or usage == 'sitw':
            max_cut_portion = Settings().network.max_test_cut_portion
        else:
            max_cut_portion = Settings().network.max_val_cut_portion
       
        index = 0
        utt = data.utterances
        #print('length of utterances:{}'.format(n_segments))
        while index < len(utt):
            fixed_segment_length = utt[index].n_signal
            if fixed_segment_length > Settings().network.max_batch_size_in_frames:
                fixed_segment_length = Settings().network.max_batch_size_in_frames
            max_segment_length = fixed_segment_length / (1 - max_cut_portion)
            frames_filled = 0
            batch = []
            while frames_filled + fixed_segment_length <= Settings().network.max_batch_size_in_frames:
                frames_filled += fixed_segment_length
                batch.append(utt[index])
                index += 1
                if index == len(utt) or utt[index].n_signal > max_segment_length:
                    break
            
            self.batch_data.append(batch)



        '''
        number_of_minibathes = math.ceil(len(data) / length)
        for _ in range(number_of_minibathes):
              
            self.batch_data.append([])
            for i in range(length):
                if index==len(utt)-1:
                    index = index-64
                else:
                    index = index+1
                self.batch_data[-1].append(utt[index])
        '''  

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, index: int):
        utt = self.batch_data[index]
        ids,spk = [],[]
        n_frame = utt[0].n_signal
        features =[]
        for i in range(len(utt)):
            if utt[i].n_signal<n_frame:
                n_frame = utt[i].n_signal
        for i in range(len(utt)):
            ids.append(utt[i].utt_id)
            spk.append(utt[i].spk_id)
            if '_noise' in ids[-1] or '_music' in ids[-1] or '_babble' in ids[-1] or '_rev' in ids[-1]:
                mix_path = ids[-1]+'_s.npz'
            else:
                parts = ids[-1].split('_')
                if len(parts)>1:
                    del(parts[-1])
                mix_path = '_'.join(parts)+'_s.npz'
            if len(self.path_list)>1:
                num = ids[-1].split('-')[0]
                num = int(num[2:])
                if num>10000:
                    idx = (num-10000)//200
                    path = os.path.join(self.path_list[0],str(idx))
                else:
                    idx = num//200
                    path = os.path.join(self.path_list[1],str(idx)) 
            else:
                path = self.path_list[0]
          #  mix_path = ids[-1]+'_mix.npz'
            mix_path = np.load(os.path.join(path, mix_path))
            mix_lps = mix_path['arr_0']
            mix_lps = torch.tensor(mix_lps)
            features.append(mix_lps[:,:n_frame])
        features = torch.stack(features)

        sample = {
            "id": ids,
            "len":n_frame,
            "spk":spk,
            "features":features
        }
        return sample
       

#def compute_losses_and_accuracies(dataloaders, network, mse, ce):
def compute_losses_and_accuracies(dataloaders, network, mse, ce, mean, std, mean_lps, std_lps):
    network.eval()
    losses_and_accuracies = []
    for dataloader in dataloaders:
        correct_clean,correct_noisy = 0,0
        segment_count = 0
        mse1_sum,mse2_sum,ce1_sum,ce2_sum = 0,0,0,0
        feature_path =  ['/data_a8/mayi/mix_speech_voxceleb/voxceleb1_fea','/data_a8/mayi/mix_speech_voxceleb/voxceleb2_fea']
        for batch in dataloader:
            speaker_labels = batch['spk']
            
            ids = batch['id']
            n_frame = batch['len']
            features, lps = extract(feature_path, ids, n_frame,'validation', False)

            features = (features-mean)/std
            features = features.float()
            lps = (lps-mean_lps)/std_lps 
            lps = lps.float()

            batch_size = len(speaker_labels)
            #speaker_labels = torch.tensor(speaker_labels)
            speaker_labels = speaker_labels.to(Settings().computing.device)
            with torch.no_grad():
                noisy_spk, clean_spk, noisy_hidden, clean_hidden, target_hidden = network(features, lps, mean_lps.to(Settings().computing.device),std_lps.to(Settings().computing.device))
                #noisy_spk, clean_spk, noisy_hidden, clean_hidden, target_hidden = network(features, lps)
                mse_noisy = torch.zeros(1).to(Settings().computing.device)
                for hidden_num in range(len(target_hidden)):
                    mse_noisy = mse_noisy+mse(noisy_hidden[hidden_num], clean_hidden[hidden_num])
                #mse_clean = torch.zeros(1).to(Settings().computing.device)
                #for hidden_num in range(len(target_hidden)):
                    #mse_clean = mse_clean+mse(clean_hidden[hidden_num], target_hidden[hidden_num])

                ce_clean = ce(clean_spk, speaker_labels)
                ce_noisy =ce(noisy_spk, speaker_labels)
            mse1_sum += mse_noisy.item()  
            #mse2_sum += mse_clean.item() 
            ce1_sum += ce_noisy.item()
            ce2_sum += ce_clean.item()
            clean_spks = torch.argmax(clean_spk, dim=1)
            noisy_spks = torch.argmax(noisy_spk, dim=1)
            correct_clean += torch.sum(clean_spks == speaker_labels)
            correct_noisy += torch.sum(noisy_spks == speaker_labels)
            segment_count += batch_size
            

        losses_and_accuracies.append((mse1_sum / segment_count, mse2_sum / segment_count, ce1_sum / segment_count, ce2_sum / segment_count, float(correct_noisy) / segment_count * 100, float(correct_clean) / segment_count * 100))
   
    network.train()
    return losses_and_accuracies

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


# The input utterance list will get sorted (in-place) according to utterance length, returned embeddings are in the sorted order
def extract_embeddings(path, data: UtteranceList, network: Type[torch.nn.Module], mean_mix, std_mix, mean_clean, std_clean, use='sitw', length=1):
#def extract_embeddings(data: UtteranceList, network: Type[torch.nn.Module], use='sitw', length=1):
    print('Extracting {} embeddings...'.format(len(data.utterances)))
    mean_mix = mean_mix.to(Settings().computing.device)
    std_mix = std_mix.to(Settings().computing.device)
    mean_clean = mean_clean.to(Settings().computing.device)
    std_clean = std_clean.to(Settings().computing.device)
 
    network.eval()
    dataloader = get_dataloader(data, path, length, usage=use)
    embeddings = torch.zeros(len(data.utterances), Settings().network.embedding_size)
    counter = 0
    start_time = time.time()
    for index, batch in enumerate(dataloader):
        ids = batch['id']
        n_frame = batch['len']
       
        features = batch['features'].squeeze(0).to(Settings().computing.device)    
        features = (features-mean_mix)/std_mix
        features = features.float()
        with torch.no_grad():
            lps, batch_embeddings = network(features,features,mean_clean, std_clean,'extract_embeddings')
            #lps, batch_embeddings = network(features,features,'extract_embeddings')
        embeddings[counter:counter+batch_embeddings.size()[0], :] = batch_embeddings
        test_finiteness(batch_embeddings, str(index))
        counter += batch_embeddings.size()[0]
        if index % (Settings().network.extraction_print_interval) == (Settings().network.extraction_print_interval) - 1:
            print('{:.0f} seconds elapsed, {}/{} batches, {}/{} utterances'.format(time.time() - start_time, index+1, len(dataloader), counter, len(data.utterances)))
    data.embeddings = embeddings

# The input utterance list will get sorted (in-place) according to utterance length, returned embeddings are in the sorted order
def extract_neural_features(data: UtteranceList, network: Type[torch.nn.Module]):
    print('Extracting neural features for {} utterances...'.format(len(data.utterances)))
    network.eval()
    dataloader = get_dataloader(data, usage='extraction')
    feature_list = []
    counter = 0
    start_time = time.time()
    for index, batch in enumerate(dataloader):
        features = batch
        features = features.to(Settings().computing.device)
        with torch.no_grad():
            batch_features = network(features, 'extract_features').transpose(1, 2)
        feature_list.extend([x.squeeze(0) for x in torch.split(batch_features, 1, dim=0)])
        counter += len(batch_features)
        if index % (Settings().network.extraction_print_interval) == (Settings().network.extraction_print_interval) - 1:
            print('{:.0f} seconds elapsed, {}/{} batches, {}/{} utterances'.format(time.time() - start_time, index+1, len(dataloader), counter, len(data.utterances)))
    data.neural_features = feature_list

# The input utterance list will get sorted (in-place) according to utterance length, returned embeddings are in the sorted order
def extract_stats(data: UtteranceList, network: Type[torch.nn.Module], second_order=True):
    print('Extracting sufficient statistics for {} utterances...'.format(len(data.utterances)))

    network.eval()
    dataloader = get_dataloader(data, usage='extraction')
    feat_dim = Settings().network.stat_size
    zeroth = torch.zeros(len(data.utterances), Settings().network.n_clusters)
    first = torch.zeros(len(data.utterances), Settings().network.n_clusters, feat_dim)
    if second_order:
        mode = 'extract_training_stats'
        second_sum = torch.zeros(Settings().network.n_clusters, feat_dim, feat_dim)
    else:
        mode = 'extract_testing_stats'  
    counter = 0
    start_time = time.time()
    for index, batch in enumerate(dataloader):
        features = batch
        features = features.to(Settings().computing.device)
        with torch.no_grad():
            stats = network(features, mode)
        z = stats[0].cpu()
        f = stats[1].cpu()
        zeroth[counter:counter+stats[0].size()[0], :] = z
        first[counter:counter+stats[1].size()[0], :, :] = f
        if second_order:
            second_sum += stats[2].cpu()
            test_finiteness(second_sum, 'second ' + str(index))
        test_finiteness(z, 'zeroth ' + str(index))
        test_finiteness(f, 'first ' + str(index))
        counter += stats[0].size()[0]
        if index % (Settings().network.extraction_print_interval) == (Settings().network.extraction_print_interval) - 1:
            print('{:.0f} seconds elapsed, {}/{} batches, {}/{} utterances'.format(time.time() - start_time, index+1, len(dataloader), counter, len(data.utterances)))
    if not second_order:
        second_sum = None
    data.stats = SufficientStats(zeroth, first, second_sum)

def _collater(batch):
    # Batch is already formed in the DataSet object (batch consists of a single element, which is actually the batch itself).
    return batch[0]

def validation_dataloader(data: UtteranceList,batch_size):
    #dataset = MaskDataset(data,'train')
    dataset = ValidationDataset(data,1)
    print('Feature loader initialized!')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2)

def get_dataloader(data,path, length=1,usage='validation'):
    dataset = TestDataset(data, length, path, usage)
    print('Feature loader for {} initialized!'.format(usage))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=Settings().computing.network_dataloader_workers)

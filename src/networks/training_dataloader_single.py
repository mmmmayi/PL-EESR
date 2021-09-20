# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).
import kaldiio
from collections import defaultdict, deque
from typing import NamedTuple
import os
import random
import math
import librosa
import numpy as np
import torch
import asvtorch.src.mask.utils as utils
from torch.utils.data import DataLoader, Dataset
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.mask.utils import generate_lps,extract
#from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
from asvtorch.src.settings.settings import Settings


class MaskDataset(Dataset):
    def __init__(self, data: UtteranceList):	
        super().__init__()
        data.sort(reverse=False)
        self.data = data
        self.batch_data = []
        self.name = data.name
        start = 0
        self.path_list = ['/data_a8/mayi/mix_speech_voxceleb/voxceleb1_fea','/data_a8/mayi/mix_speech_voxceleb/voxceleb2_fea']
        #number_of_minibathes = math.ceil(Settings().network.utts_per_speaker_in_epoch * len(self.data) / Settings().network.minibatch_size)
        number_of_minibathes = math.ceil(len(self.data) / 1)
        index = -1
        num = 0
        for _ in range(number_of_minibathes):
            self.batch_data.append([])
            for _ in range(1):
                if index==len(self.data)-1:
                    index = index-64
                    num += 1
                else:
                    index = index+1
                #print('length of utt_temp:{},index:{}'.format(len(self.data),index))

                self.batch_data[-1].append(self.data.utterances[index])
        #print('debug mask dataset, length of train dataset:{}, length of train dataset[0]:{}'.format(len(self.batch_data),len(self.batch_data[0])))
        print('broken time:{}'.format(num))

    def shuffle(self, seed):
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(len(self.batch_data)))
        temp_data = [self.batch_data[i] for i in indices]
        self.batch_data = temp_data
        
    def __len__(self):
        return len(self.batch_data)

    def _clip_point(self, frame_len):
        clip_length = np.random.randint(Settings().network.min_clip_size, Settings().network.max_clip_size + 1)
        
        start_point = 0
        end_point = frame_len
        if frame_len>=clip_length:
            start_point = np.random.randint(0, frame_len - clip_length + 1)
            end_point = start_point + clip_length
        return start_point, end_point

    def __getitem__(self, index):
        '''
        for utt in self.batch_data[index]:
            mix_path = utt.utt_id+'_mix.wav'
            mix_path = os.path.join(path, mix_path)
            clean_path = utt.utt_id+'_s.wav'
            clean_path = os.path.join(path, clean_path)
            
            mix_lps = generate_lps(mix_path)
            clean_lps = generate_lps(clean_path)
#            mix_lps = np.load(os.path.join(path,self.name,'lists/mix_lps', utt.utt_id+'.npz'))
#            mix_lps = mix_lps['arr_0']
#            clean_lps = np.load(os.path.join(path,self.name,'lists/clean_lps', utt.utt_id+'.npz'))
#            clean_lps = clean_lps['arr_0']
            spk = utt.spk_id

        features.append(mix_lps[start_point:end_point,:])
        target_lps.append(clean_lps[start_point:end_point,:])
        spk_id.append(spk)

        features = torch.stack(features).permute(0,2,1)
        target_lps = torch.stack(target_lps)
        print('features:{},target_lps:{}'.format(features.device,target_lps.device))
        return features, target_lps, torch.LongTensor(spk_id)
        '''
        utt = self.batch_data[index][0]
        n_frame = utt.n_signal
        start_point, end_point = self._clip_point(n_frame)
        spk = utt.spk_id
        id = utt.utt_id
        num = id.split('-')[0]
        num = int(num[2:])
        if num>10000:
            idx = (num-10000)//200
            path = os.path.join(self.path_list[0],str(idx))
        else:
            idx = num//200
            path = os.path.join(self.path_list[1],str(idx))
        mix_path = id+'_mix.npz'
        mix_path = np.load(os.path.join(path, mix_path))
        mix_lps = mix_path['arr_0']
        mix_lps = torch.tensor(mix_lps[:,start_point:end_point])
        if '_noise' in id or '_music' in id or '_babble' in id:
            parts = id.split('_')
            del(parts[-1])
            del(parts[-1])
            clean_name = '_'.join(parts)
        elif '_rev' in id:
            clean_name = id
        else:
            parts = id.split('_')
            del(parts[-1])
            clean_name = '_'.join(parts)
        clean_path = clean_name+'_s.npz'
        clean_path = np.load(os.path.join(path, clean_path))
        clean_lps = clean_path['arr_0']
        clean_lps = torch.tensor(clean_lps[:,start_point:end_point])
        sample = {
            "id": utt.utt_id,
            "spk": spk,
            "len":utt.n_signal,
            "features":mix_lps,
            "clean_lps":clean_lps
        }
        return sample

class SpeakerData(NamedTuple):
    utterance_list: UtteranceList
    utterance_queue: deque


def _select_next_clip(speaker_data: SpeakerData, clip_length: int): ## REMEMBER VAD TOLERANCE
    for _ in range(len(speaker_data.utterance_list.utterances) + len(speaker_data.utterance_queue)):
        if not speaker_data.utterance_queue:
            shuffled_utts = speaker_data.utterance_list.utterances.copy()
            random.shuffle(shuffled_utts)
            speaker_data.utterance_queue.extend(shuffled_utts)
        utt = speaker_data.utterance_queue.popleft()
        frame_count = utt.n_signal
        if frame_count >= clip_length:
            start_point = np.random.randint(0, frame_count - clip_length + 1)
            end_point = start_point + clip_length
            return utt, (start_point, end_point)
    print('Warning: speaker "{}" does not have training utterances of length {} or longer. Skipping!'.format(speaker_data.utterance_list.name, clip_length))
    return None, None

def get_global_mean_variance(dataloader):
    clean_mean, noisy_mean = 0.0,0.0
    clean_variance,noisy_variance = 0.0,0.0
    N_ = 0
    feature_path = '/data_a8/mayi/mix_speech_voxceleb/voxceleb1_mask_aug'
    print("Calculating traindataset mean and variance...")
    for batch in dataloader:
    #for feat, _ in dataloader:
        
        ids = batch['id']
        n_frame = batch['len']
        features, lps = extract(feature_path, ids, n_frame,usage='validation', clip=False)
        #features = features.transpose(1,2)
        #lps= lps.transpose(1,2)
        #print(features.shape)#[batch,201,503]
        #print(lps.shape)#[batch,201,503]
            
            
            #print(feat.shape)
        #print(N_)
        N_ += features.shape[0]
        noisy_mean += features.mean(2).sum(0)
        noisy_variance += features.var(2).sum(0)
        clean_mean += lps.mean(2).sum(0)
        clean_variance += lps.var(2).sum(0)
        #print(clean_mean.shape) #[201]
#        print('feat shape:{},N:{},mean.shape:{}'.format(feat.shape,N_,mean.shape))
    clean_mean = clean_mean / N_
    clean_variance = clean_variance / N_
    noisy_mean = noisy_mean / N_
    noisy_variance = noisy_variance / N_
 
    standard_clean_dev = torch.sqrt(clean_variance)
    standard_noisy_dev = torch.sqrt(noisy_variance)
    return clean_mean, standard_clean_dev, noisy_mean, standard_noisy_dev

def _collater(batch):
    # Batch is already formed in the DataSet object (batch consists of a single element, which is actually the batch itself).
    return batch[0]

def get_dataloader(data: UtteranceList,batch_size):
    #dataset = MaskDataset(data,'train')
    dataset = MaskDataset(data)
    print('Feature loader initialized!')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2)
    #return MaskDataloader(mode='train',dataset=dataset, batch_size=1, shuffle=False, num_workers=Settings().computing.network_dataloader_workers)


# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).
import kaldiio
from collections import defaultdict, deque
from typing import NamedTuple
import random
import math
import numpy as np
import torch
import asvtorch.src.mask.utils as utils
from torch.utils.data import DataLoader, Dataset
import asvtorch.src.mask
from asvtorch.src.utterances.utterance_list import UtteranceList
from torch.utils.data.distributed import DistributedSampler
#from asvtorch.src.frontend.featureloaders.featureloader import FeatureLoader
from asvtorch.src.settings.settings import Settings

class SpeakerData(NamedTuple):
    utterance_list: UtteranceList
    utterance_queue: deque


def _collate(batch):
    return batch[0]

class MFCCTrainingDataset(Dataset):
    def __init__(self, data: UtteranceList):
        super().__init__()
        data.sort()
        self.data = data
        self.batch_data = []
        start = 0
        number_of_minibathes = math.ceil(len(self.data) / Settings().network.minibatch_size)
        index = -1
        num = 0
        #number_of_minibathes = math.ceil(Settings().network.utts_per_speaker_in_epoch * len(self.data) / Settings().network.minibatch_size)
        for _ in range(number_of_minibathes):
            self.batch_data.append([])
            
            for _ in range(Settings().network.minibatch_size):
#            for _ in range(2):
                if index==len(self.data)-1:
                    index = index-64
                    num+=1
                else:
                    index = index+1
                #print('length of utt_temp:{},index:{}'.format(len(self.data),index))
                self.batch_data[-1].append(self.data.utterances[index])

          
        #print('debug mask dataset, length of train dataset:{}, length of train dataset[0]:{}'.format(len(self.batch_data),len(self.batch_data[0])))

    def shuffle(self, seed):
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(len(self.batch_data)))
        temp_data = [self.batch_data[i] for i in indices]
        self.batch_data = temp_data

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, index):

        features,target_lps,spk_id = [],[],[]
        fix_length = self.batch_data[index][0].n_signal
        fix_length = min(8000,fix_length)
        for utt in self.batch_data[index]:
            mix_lps = kaldiio.load_mat(utt.mix_lps)
#            print('lps shape:{}'.format(mix_lps[:fix_length,:].shape))
            #noise_lps = kaldiio.load_mat(utt.noise_lps)
            clean_lps = kaldiio.load_mat(utt.clean_lps)
            spk = utt.spk_id

            #irm = utils.lps2irm(noise_lps, clean_lps, window=np.hamming(400), nperseg=400, noverlap=240)
            features.append(mix_lps[:fix_length,:])
            target_lps.append(clean_lps[:fix_length,:])
            spk_id.append(spk)
        features = torch.from_numpy(np.transpose(np.dstack(features), (2, 1, 0)))
        target_lps = torch.from_numpy(np.transpose(np.dstack(target_lps), (2, 0, 1)))
        return features, target_lps, torch.LongTensor(spk_id)

def get_global_mean_variance(dataloader,mode='training'):
    mean = 0.0
    variance = 0.0
    N_ = 0

    print("Calculating traindataset mean and variance...")
    for batch in dataloader:
    #for feat, _ in dataloader:
        if mode=='extraction':
            feat = batch
        else:
            _,feat,_ = batch
            feat = feat.transpose(1,2)
            #print(feat.shape)
        #print(N_)
        N_ += feat.size(0)
        mean += feat.mean(2).sum(0)
        variance += feat.var(2).sum(0)
#        print('feat shape:{},N:{},mean.shape:{}'.format(feat.shape,N_,mean.shape))
    mean = mean / N_
    variance = variance / N_
    standard_dev = torch.sqrt(variance)

    return mean, standard_dev

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


def get_dataloader(data: UtteranceList):
    dataset = MFCCTrainingDataset(data)
    print('Feature loader initialized!')
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=Settings().computing.network_dataloader_workers, collate_fn=_collate, sampler=DistributedSampler(dataset))

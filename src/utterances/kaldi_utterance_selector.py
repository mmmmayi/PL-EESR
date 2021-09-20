# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import List, Optional, Tuple
import random
import os
import re
import sys

#import kaldi.util.table as ktable
import numpy as np

from kaldiio import ReadHelper

from asvtorch.src.settings.settings import Settings
from asvtorch.src.frontend.frame_selector import FrameSelector
from asvtorch.src.utterances.utterance import Utterance_mask
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.misc.ordered_set import OrderedSet
from asvtorch.src.utterances.abstract_utterance_selector import AbstractUtteranceSelector
import asvtorch.src.misc.fileutils as fileutils


class MaskUtteranceSelector(AbstractUtteranceSelector):

    def choose_all(self, dataset: str) -> UtteranceList:
        utterance_list = _choose_utterances(dataset, None)
        utterance_list.name = dataset
        return utterance_list

    def choose_longest(self, dataset: str, n: int) -> UtteranceList:
        utt2len_file = _get_file(dataset, 'utt2featSize')
        selected_utts = OrderedSet()
        utts = []
        n_signal = []
        with open(utt2len_file) as f:
            for line in f:
                parts = line.split()
                utts.append(parts[0])
                n_signal.append(int(parts[1].strip()))
        n_signal = np.asarray(n_signal, dtype=int)
        indices = np.argsort(n_signal)
        indices = indices[-n:]
        for index in indices:
            selected_utts.add(utts[index])
        utterance_list = _choose_utterances(dataset, selected_utts)
        utterance_list.name = '{}_{}_longest'.format(dataset, n)
        return utterance_list

    def choose_random(self, dataset: str, n: int, seed: int = 0) -> UtteranceList:
        random.seed(seed)
        utt2len_file = _get_file(dataset, 'utt2featSize')
        utts = []
        with open(utt2len_file) as f:
            for line in f:
                parts = line.split()
                utts.append(parts[0])
        utterance_list = _choose_utterances(dataset, OrderedSet(random.sample(utts, n)))
        utterance_list.name = '{}_{}_random_seed_{}'.format(dataset, n, seed)
        return utterance_list

    def choose_from_list(self, dataset: str, id_list: List[str]) -> UtteranceList:
        utterance_list = _choose_utterances(dataset, OrderedSet(id_list))
        utterance_list.name = '{}_from_list'.format(dataset)
        return utterance_list

    def choose_from_trials(self, dataset: str, trial_file: str, side: str = 'both') -> UtteranceList:  # both, enroll, or test     
        assert side in ['both', 'enroll', 'test']
        list_folder = fileutils.get_list_folder(dataset)
        trial_file = os.path.join(list_folder, trial_file)
        selected_utts = OrderedSet()
        with open(trial_file) as f:
            for line in f: # Remove duplicates, keep the insertion order
                parts = line.split()
                if side == 'both':
                    selected_utts.add(parts[0])
                    selected_utts.add(parts[1])
                elif side == 'enroll':
                    selected_utts.add(parts[0])
                elif side == 'test':
                    selected_utts.add(parts[1])
        utterance_list = _choose_utterances(dataset, selected_utts)
        utterance_list.name = '{}_{}'.format(dataset, trial_file.rsplit('.', 1)[0])
        return utterance_list

    def choose_regex(self, dataset: str, regex: str) -> UtteranceList:  # utt or spk

        #utt2len = _get_file(dataset, 'utt2len')
        utt2len = _get_file(dataset, 'utt2featSize')
        selected_utts = OrderedSet()
        with open(utt2len) as f:
            for line in f:
                #print(line)
                parts = line.split()

                current_id = parts[0]
                if re.search(regex, current_id) is not None:
                    #print('current_id:',current_id)
                    selected_utts.add(current_id)
        utterance_list = _choose_utterances(dataset, selected_utts)
        #utterance_list.name = dataset + '_regex'
        return utterance_list

    def choose_random_regex(self, dataset: str, n: int, regex: str, seed: int = 0) -> UtteranceList:
        random.seed(seed)
        utt2len = _get_file(dataset, 'utt2featSize')
        utts = []
        with open(utt2len) as f:
            for line in f:
                current_id = parts[0]
                if re.search(regex, current_id) is not None:
                    utts.append(current_id)
        utterance_list = _choose_utterances(dataset, OrderedSet(random.sample(utts, n)))
        utterance_list.name = '{}_{}_random_seed_{}'.format(dataset, n, seed) + '_regex'
        return utterance_list

def _get_file(dataset: str, filename: str) -> str:
    list_folder = fileutils.get_list_folder(dataset)
    return os.path.join(list_folder, filename)

def _choose_utterances(dataset: str, selected_utts: Optional[OrderedSet]) -> UtteranceList:
    # content of OrderSet need to be utt_id of wav files
    #datafile = _get_file(dataset, 'utt2len_forSV')
    datafile = _get_file(dataset, 'mix_utt2spk')
    size_file = _get_file(dataset,'mix_utt2featSize')
    f = open(size_file,'r')
    line = f.readline()
    shape = {}

    while line:
        parts = line.split(' ')
        
        shape[parts[0]]=int(parts[1].strip('\n'))
        #print(shape[parts[0]])
        line = f.readline()
    f.close()

    if selected_utts is not None:
        utts = []
        #print('selected_utts is not None')

        for selected_utt in selected_utts:
            #print(selected_utt)
            utts.append(selected_utt.replace('.wav',''))
#    print('utts length:',len(utts))

    utterances = []
    f = open(datafile,'r')
    line = f.readline()
    #print('utts:',utts)
    while line:
        #print(line)
        parts = line.split(' ')
        path = parts[0]
        spk_id = parts[1]
        utt_id = parts[2].strip('\n')
        
        #print('utt_id:{}'.format(utt_id))
        if selected_utts is None or utt_id in utts:
            n_signal = shape[utt_id]
             
            utterances.append(Utterance_mask(utt_id, spk_id, n_signal))

        line = f.readline()
    
    f.close()
    return UtteranceList(utterances,name=dataset)



'''
class KaldiUtteranceSelector(AbstractUtteranceSelector):

    def choose_all(self, dataset: str) -> UtteranceList:
        utterance_list = _choose_utterances(dataset, None)
        utterance_list.name = dataset
        return utterance_list

    def choose_longest(self, dataset: str, n: int) -> UtteranceList:
        utt2num_frames_file = _get_file(dataset, 'utt2num_frames')
        selected_utts = OrderedSet()
        utts = []
        num_frames = []
        with open(utt2num_frames_file) as f:
            for line in f:
                parts = line.split()
                utts.append(parts[0])
                num_frames.append(int(parts[1].strip()))
        num_frames = np.asarray(num_frames, dtype=int)
        indices = np.argsort(num_frames)
        indices = indices[-n:]
        for index in indices:
            selected_utts.add(utts[index])
        utterance_list = _choose_utterances(dataset, selected_utts)
        utterance_list.name = '{}_{}_longest'.format(dataset, n)
        return utterance_list

    def choose_random(self, dataset: str, n: int, seed: int = 0) -> UtteranceList:
        random.seed(seed)
        utt2spk_file = _get_file(dataset, 'utt2spk')
        utts = []
        with open(utt2spk_file) as f:
            for line in f:
                parts = line.split()
                utts.append(parts[0])
        utterance_list = _choose_utterances(dataset, OrderedSet(random.sample(utts, n)))
        utterance_list.name = '{}_{}_random_seed_{}'.format(dataset, n, seed)
        return utterance_list

    def choose_from_list(self, dataset: str, id_list: List[str]) -> UtteranceList:
        utterance_list = _choose_utterances(dataset, OrderedSet(id_list))
        utterance_list.name = '{}_from_list'.format(dataset)
        return utterance_list

    def choose_from_trials(self, dataset: str, trial_file: str, side: str = 'both') -> UtteranceList:  # both, enroll, or test     
        assert side in ['both', 'enroll', 'test']
        list_folder = fileutils.get_list_folder(dataset)
        trial_file = os.path.join(list_folder, trial_file)
        selected_utts = OrderedSet()
        with open(trial_file) as f:
            for line in f: # Remove duplicates, keep the insertion order
                parts = line.split()
                if side == 'both':
                    selected_utts.add(parts[0])
                    selected_utts.add(parts[1])
                elif side == 'enroll':
                    selected_utts.add(parts[0])
                elif side == 'test':
                    selected_utts.add(parts[1])
        utterance_list = _choose_utterances(dataset, selected_utts)
        utterance_list.name = '{}_{}'.format(dataset, trial_file.rsplit('.', 1)[0])
        return utterance_list

    def choose_regex(self, dataset: str, regex: str, id_type: str = 'utt') -> UtteranceList:  # utt or spk
        assert id_type == 'utt' or id_type == 'spk'
        utt2spk_file = _get_file(dataset, 'utt2spk')
        selected_utts = OrderedSet()
        with open(utt2spk_file) as f:
            for line in f:
                parts = line.split()
                if id_type == 'utt':
                    current_id = parts[0]
                else:
                    current_id = parts[1].strip()
                if re.search(regex, current_id) is not None:
                    selected_utts.add(parts[0])
        utterance_list = _choose_utterances(dataset, selected_utts)
        utterance_list.name = dataset + '_regex'
        return utterance_list

    def choose_random_regex(self, dataset: str, n: int, regex: str, seed: int = 0) -> UtteranceList:
        random.seed(seed)
        utt2spk_file = _get_file(dataset, 'utt2spk')
        utts = []
        with open(utt2spk_file) as f:
            for line in f:
                parts = line.split()
                if re.search(regex, parts[0]) is not None:
                    utts.append(parts[0])
        utterance_list = _choose_utterances(dataset, OrderedSet(random.sample(utts, n)))
        utterance_list.name = '{}_{}_random_seed_{}'.format(dataset, n, seed) + '_regex'
        return utterance_list
'''
def _get_file(dataset: str, filename: str) -> str:
    list_folder = fileutils.get_list_folder(dataset)
    return os.path.join(list_folder, filename)

def _get_kaldi_dataset_files(dataset: str) -> Tuple[str, str, str, str]:
    list_folder = fileutils.get_list_folder(dataset)
    feat_scp_file = os.path.join(list_folder, 'feats.scp')
    vad_scp_file = os.path.join(list_folder, 'vad.scp')
    temp_vad_scp_file = os.path.join(list_folder, 'temp_vad.scp')
    utt2spk_file = os.path.join(list_folder, 'utt2spk')
    return feat_scp_file, vad_scp_file, temp_vad_scp_file, utt2spk_file
'''
def _choose_utterances(dataset: str, selected_utts: Optional[OrderedSet]) -> UtteranceList:
    if selected_utts is not None:
        selected_utts_splitted = []
        selected_utts_multi = []
        for selected_utt in selected_utts:
            split_result = selected_utt.split('&')
            selected_utts_splitted.extend(split_result)
            selected_utts_multi.append(split_result)
        selected_utts_splitted = set(selected_utts_splitted)
    utt2index = {}
    feat_scp_file, vad_scp_file, temp_vad_scp_file, utt2spk_file = _get_kaldi_dataset_files(dataset)
    data_folder = Settings().paths.output_folder
    base_folder = os.sep + os.path.basename(os.path.normpath(data_folder)) + os.sep
    feat_rxfilenames = []
    utts = []
    spks = []
    with open(feat_scp_file) as f1, open(vad_scp_file) as f2, open(utt2spk_file) as f3, open(temp_vad_scp_file, 'w') as out_vad:
        for line1, line2, line3 in zip(f1, f2, f3):
            parts1 = line1.split()
            if selected_utts is None or parts1[0] in selected_utts_splitted:
                utt2index[parts1[0]] = len(utts)
                parts2 = line2.split()
                parts3 = line3.split()
                if parts1[0] != parts2[0] or parts1[0] != parts3[0]:
                    sys.exit('Error: scp-files are not aligned!')
                feat_loc = parts1[1].split(base_folder)[1].strip()
                vad_loc = parts2[1].split(base_folder)[1].strip()
                feat_rxfilenames.append(os.path.join(data_folder, feat_loc))
                vad_rxfilename = os.path.join(data_folder, vad_loc)
                out_vad.write('{} {}\n'.format(parts1[0], vad_rxfilename))
                utts.append(parts1[0])
                spks.append(parts3[1].strip())
    temp_vad_scp_file = 'scp:' + temp_vad_scp_file
    reader = ktable.SequentialVectorReader(temp_vad_scp_file)
    frame_selectors = []
    for _, value in reader:
        boolean_selectors = value.numpy().astype(bool)
        if not Settings().features.sad_enabled:
            boolean_selectors[:] = True
        frame_selectors.append(FrameSelector(boolean_selectors))
    utterances = []
    if selected_utts is None:
        for feat, vad, utt, spk in zip(feat_rxfilenames, frame_selectors, utts, spks):
            utterances.append(Utterance(feat, vad, utt, spk))        
    else:
        for multi_utt in selected_utts_multi:
            feats = []
            vads = []
            index = utt2index[multi_utt[0]]
            utt_ids = []
            spk = spks[index]
            for utt in multi_utt:
                index = utt2index[utt]
                utt_ids.append(utts[index])
                feats.append(feat_rxfilenames[index])
                vads.append(frame_selectors[index])
            if len(feats) == 1:
                feats = feats[0]
                vads = vads[0]
            utt = '&'.join(utt_ids)
            utterances.append(Utterance(feats, vads, utt, spk))      
    return UtteranceList(utterances)
'''

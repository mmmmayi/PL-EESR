from kaldiio import WriteHelper
import os
import numpy as np
import librosa
import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.settings.settings import Settings
from asvtorch.src.mask.utils import read_wav,mix_waveform, save_wav_files,extract_lps_fea,generate_lps
from asvtorch.src.utterances.utterance_list import UtteranceList
import random
from random import choice

def _get_file(dataset: str, filename: str) -> str:
    list_folder = fileutils.get_list_folder(dataset)
    return os.path.join(list_folder, filename)
    
def MFCC(fileload):
    utt2spk = _get_file(fileload,'utt2spk')
    spks = open(utt2spk,'r')
    line = spks.readline()
    f = open(_get_file(fileload,'utt2featSize'),'w')
    
    with WriteHelper('ark,scp:'+_get_file(fileload,'feature.ark')+','+_get_file(fileload,'feature.scp')) as writer:
        print('start computing feature') 
        while line:
            parts = line.split(' ')
            path = parts[0]
            spk_id = parts[1]
            utt_id = parts[2]
            sr,y = read_wav(path)
            y = y.astype(float)
            S = librosa.feature.mfcc(y=y, sr=Settings().network.sr,n_mfcc=30, n_fft=Settings().network.nperseg, hop_length=Settings().network.noverlap, norm='ortho', win_length=None, window='hamming', center=False, power=2.0, fmin=0,fmax=Settings().network.sr/2)
#            print('test for spk_id:',utt.spk_id.strip('\n'))
#            return
            writer(utt_id.strip('\n'),S)
            f.write(utt_id.strip('\n')+' '+str(S.shape[1])+'\n')
            line = spks.readline()
            


def mix_audio_feature(fileload):
    input_list_speech = _get_file(fileload,'utt2spk')
    input_list_noise = _get_file(fileload,'noise_train')
    snr = random.randint(0,20)
    
    with open(input_list_noise,'r') as f:
        noise_list = f.readlines()
    noise_num = len(noise_list)
    size = open(_get_file(fileload,'mix_utt2featSize'),'w')
    f = open(input_list_speech, 'r')
    line = f.readline()
    lists = '/data_a8/mayi/mix_speech_voxceleb/'+fileload
    output_dir = '/data_a8/mayi/mix_speech_voxceleb/'+fileload
#    with WriteHelper('ark,scp:'+_get_file(fileload,'mix_lps.ark')+','+_get_file(fileload,'mix_lps.scp')) as mix_writer,WriteHelper('ark,scp:'+_get_file(fileload,'speech_lps.ark')+','+_get_file(fileload,'speech_lps.scp')) as speech_writer,WriteHelper('ark,scp:'+_get_file(fileload,'noise_lps.ark')+','+_get_file(fileload,'noise_lps.scp')) as noise_writer, open(_get_file(fileload,'mix_utt2spk'),'w') as mix_utt2spk:
    with open(_get_file(fileload,'mix_utt2spk'),'w') as mix_utt2spk:
        for path in os.listdir(lists):
        #while line:
            #rand = np.random.randint(0, noise_num, size=1)[0]
            #parts = line.split(' ')
            #print(parts)
            #inputfile1 = parts[0]
            #utt_id = parts[2].strip('\n')
            #inputfile2 = noise_list[rand].strip('\n')
            #spk_id = parts[1]
#            (mixed_audio, speech_audio, noise_audio, noise_onset, noise_offset) =mix_waveform (inputfile1, inputfile2, snr)
#            path = save_wav_files(output_dir, utt_id, mixed_audio, speech_audio, noise_audio)
            #print(path)
#            mixed_lps, speech_lps, noise_lps = extract_lps_fea (mixed_audio, speech_audio,noise_audio)
            if '_mix' in path:
                mix_path = os.path.join(lists,path)
                mixed_lps = generate_lps(mix_path)
                utt_id = path.replace('_mix.wav','')
                spk_id = utt_id.split('-')
                spk_id = spk_id[0]
                size.write(utt_id+' '+str(mixed_lps.shape[1])+'\n')
            
                mix_utt2spk.write(mix_path+' '+spk_id+' '+utt_id+'\n')
            #mix_writer(utt_id,mixed_lps)
            #speech_writer(utt_id,speech_lps)
            #noise_writer(utt_id,noise_lps)

            #line = f.readline()



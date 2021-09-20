#-*-coding: utf-8-* 
import os
import numpy as np 
import argparse 
import csv 
import time 
import random
from scipy import signal 
#import h5py
#from sklearn import preprocessing 
import pdb
import scipy.io.wavfile as wav_io 
import random 
from random import choice
from random import sample
import csv
import shutil
from utils import mix_waveform, mix_waveform_given_NoiseInfo, save_wav_files, extract_lps_fea, save_feature_in_htkformat, save_augment, read_wav

def select(rate):
    path = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/mask/augment'

    select_path = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/mask/augment_select'
    wavs = os.listdir(path)
    n_wavs = len(wavs)/3
    n_select = round(n_wavs*rate)
    print(n_select)
    samples = sample(wavs,n_select)
    for name in samples:
        #print(name)
        #print(os.path.join(selsct_path,name))
        shutil.copy(os.path.join(path,name),os.path.join(select_path,name))

def make_augment(args):
    input_path_speech = args.speech_path # speech file list
    input_list_noise = args.noise_file # noise file list
    workdir = os.path.abspath (args.workplace)
    snr = args.snr
    repeate = choice(args.repeate)
    noise_type = args.noise_type
    output_dir= os.path.join (workdir, 'augment')
    #if not os.path.exists (output_dir):
        #os.makedirs (output_dir)

    with open (input_list_noise, 'r') as f:
        noise_list=f.readlines()
    noise_num= len (noise_list)


    for i in range(repeate): 
        for wav in os.listdir(input_path_speech):
            
            rand=np.random.randint(0, noise_num, size=1)[0]
            if i==0:
                inputfile1=os.path.join(input_path_speech,wav)
            else:
                inputfile1=os.path.join(output_dir,noise_type+wav)
            inputfile2=noise_list[rand].strip('\n')
            name = os.path.join(output_dir,noise_type+wav)
            if not os.path.exists(name):
                (mixed_audio, speech_audio, noise_audio, noise_onset, noise_offset) = mix_waveform(inputfile1, inputfile2, choice(snr))
                n_mixed = len(mixed_audio)
                if n_mixed>1:
                    save_augment(name, mixed_audio)   

def GenUtt2len(args):
    workdir = os.path.abspath (args.workplace)
    f = open(os.path.join(workdir,'lists','utt2len_forSV'),'a+')
    
    folder_aug = os.path.join(workdir,'augment_select')
    folder_raw = args.speech_path
    for folder in [folder_raw,folder_aug]:
        for wav in os.listdir(folder):
            #if '_mix.wav' in wav:
            sr, sig = read_wav(os.path.join(folder,wav))
                #print(os.path.join(folders,wav))
            f.write(os.path.join(folder,wav)+' '+str(len(sig))+'\n')

def mix_audio_normal(args):
    input_path_speech = args.speech_path # speech file list 
    input_list_noise = args.noise_file # noise file list 
    workdir = os.path.abspath (args.workplace)
    snr=args.snr

    #if not os.path.exists(workdir):
        #os.makedirs(workdir)

    output_dir= os.path.join (workdir, 'mixed_speech') + '/snr'+ str(snr) 
    #if not os.path.exists (output_dir):
        #os.makedirs (output_dir)

    with open (input_list_noise, 'r') as f:
        noise_list=f.readlines() 
    noise_num= len (noise_list)

    #fea_mixed_scp =open(scp_dir +'/SNR'+ str(snr) +'_'+ str(start_id)+'_' +str(end_id) + '_fea_mixed.list','w') 
    i = 0
    f = open(os.path.join(workdir,'utt2len'),'w')
    for wav in os.listdir(input_path_speech):
        speaker_id = wav.strip('.wav') #id0123-xxxxx
        rand2=np.random.randint(0, noise_num, size=1)[0]
        inputfile1=os.path.join(input_path_speech,wav)
        inputfile2=noise_list [rand2].strip('\n')

        path = os.path.join(output_dir,speaker_id)+'_mix.wav'
        if not os.path.exists(path):
    
            (mixed_audio, speech_audio, noise_audio, noise_onset, noise_offset) = mix_waveform(inputfile1, inputfile2, snr)
            n_mixed = len(mixed_audio)
            if n_mixed>1:
                save_wav_files(output_dir, speaker_id, mixed_audio, speech_audio, noise_audio)
                f.write(path+' '+str(n_mixed)+'\n')

        i+=1

def mix_audio_progrssive_SNRS(args):
    input_list_speech=args.speech_file # speech file list 
    input_list_noise= args.noise_file # noise file list
    workdir = os.path.abspath(args.workplace)
    start_id = args.start_id
    end_id = args.end_id
    snr = args.initial_snr
    PL_snr_gap = args.PL_snr_gap
    PL_stages_num = args.PL_stages_num

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    scp_dir =os.path.join(workdir, 'scp') 
    if not os.path.exists(scp_dir):
        os.makedirs(scp_dir)

    assert os.path.exists(input_list_speech), 'input_list_speech is not valid'
    source_speech_list= os.path.join(scp_dir)+'/source_speech.list'   
    os.system('cp %s %s' % (input_list_speech, source_speech_list))

    source_noise_list = scp_dir +'/source_noise.list'
    os.system ('cp %s %s' %(input_list_noise, source_noise_list))


    output_dir = os.path.join (workdir, 'mixed_speech')+'/snr'+str(snr)
    if not os.path.exists (output_dir):
            os.makedirs(output_dir)

    #make different stages in progressive-learning 
    for j in range (PL_stages_num):
        intermediate_snr= (j+1)* PL_snr_gap 
        output_PL_dir = os.path.join(workdir, 'mixed_speech')+ '/snr'+ str(snr+intermediate_snr)
        if not os.path.exists(output_PL_dir):
            os.makedirs(output_PL_dir)

    with open (source_speech_list, 'r') as f:
        speech_list=f.readlines()
    speech_num=len(speech_list)

    with open(source_noise_list, 'r') as f:
        noise_list=f.readlines()
    noise_num= len(noise_list)

    fea_mixed_scp= open (scp_dir +'/SNR' + str(snr) +'_seq'+str(start_id) +'_to_seq'+str(end_id) + '_fea_mixed.list', 'w') 

    for i in range (start_id, end_id):
        rand1 = np.random.randint(0, speech_num, size=1)[0] 
        rand2 = np.random.randint(0, noise_num, size=1)[0] 
        inputfile1= speech_list[rand1].strip('\n') 
        inputfile2= noise_list[rand2].strip('\n')

        # make data of the initial-SNR stage
        output_dir= os.path.join(workdir,'mixed_speech') + '/snr' + str(snr)
        (mixed_audio, speech_audio, noise_audio, noise_onset, noise_offset) =mix_waveform (inputfile1, inputfile2, snr)
        save_wav_files(output_dir, i, mixed_audio, speech_audio, noise_audio)
        #For log-power-spectrum (lps) feature, the frame length and shift are set to 32 ms and 16 ms
        mixed_lps, speech_lps, noise_lps =extract_lps_fea(mixed_audio, speech_audio, noise_audio)
        flag= save_feature_in_htkformat (output_dir, i, mixed_lps, speech_lps, noise_lps)
        if flag == True:
            fea_mixed_scp.write(os.path.join(output_dir, str(i)) + '_mix.fea\n')


        # make data of progressive-SNRS stages 
        for j in range(PL_stages_num):
            intermediate_snr= (j+1) *PL_snr_gap
            output_PL_dir =os.path.join(workdir, 'mixed_speech')+'/snr'+ str(snr+intermediate_snr)
            (mixed_audio, speech_audio, noise_audio) = mix_waveform_given_NoiseInfo(inputfile1, inputfile2, intermediate_snr, noise_onset, noise_offset)
            save_wav_files(output_PL_dir, i, mixed_audio, speech_audio, noise_audio)
            mixed_lps, speech_lps, noise_lps= extract_lps_fea(mixed_audio, speech_audio, noise_audio)
            flag=save_feature_in_htkformat(output_PL_dir, i, mixed_lps, speech_lps, noise_lps)

    fea_mixed_scp.close ()



if __name__ == '__main__' : 
    parser = argparse.ArgumentParser(description='Process of adding additive noise.')
    subparsers= parser.add_subparsers(dest='mode')

    mix_speech_normal =subparsers.add_parser('mix_speech_fixedSNR')
    mix_speech_normal.add_argument('--speech_path', type=str, required= True)
    mix_speech_normal.add_argument('--noise_file', type=str, required=True) 
    mix_speech_normal.add_argument('--workplace', type=str, required=True)
    mix_speech_normal.add_argument('--snr', type=int, required=True)

    mix_speech_ProgressiveSNR =subparsers.add_parser('mix_speech_ProgressiveSNR')
    mix_speech_ProgressiveSNR.add_argument('--speech_file', type=str, required=True) 
    mix_speech_ProgressiveSNR.add_argument('--noise_file', type=str, required=True) 
    mix_speech_ProgressiveSNR.add_argument('--workplace', type=str, required= True) 
    mix_speech_ProgressiveSNR.add_argument('--start_id', type=int, required= True )
    mix_speech_ProgressiveSNR.add_argument('--end_id', type=int, required=True) 
    mix_speech_ProgressiveSNR.add_argument('--initial_snr', type=int, required=True) 
    mix_speech_ProgressiveSNR.add_argument('--PL_snr_gap', type=int, required=True) 
    mix_speech_ProgressiveSNR.add_argument('--PL_stages_num', type=int, required=True)
 
    augment = subparsers.add_parser('augment')
    augment.add_argument('--speech_path', type=str, required= True)
    augment.add_argument('--noise_file', type=str, required=True)
    augment.add_argument('--workplace', type=str, required=True)
    augment.add_argument('--snr', nargs='+', type=int, required=True)
    augment.add_argument('--repeate', nargs='+', type=int, required=True)
    augment.add_argument('--noise_type', type=str, required=True)


    params = parser.parse_args()

    if params.mode =='mix_speech_fixedSNR':
        mix_audio_normal(params)
        print('Adding noise with {} dB, done!!!! \n'.format(params.snr))

    if params.mode == 'mix_speech_ProgressiveSNR':
        mix_audio_progrssive_SNRS(params)
        print('Adding noise with progressive SNRs, done \n')

    if params.mode == 'augment':
        #print(params)
        #make_augment(params)
        
        #select(0.809)
        GenUtt2len(params)
#python 0_addnoise.py mix_speech_fixedSNR --speech_file ./speech.list --noise_file  ./noise.list --workplace ./output_dir --start_id  0 --end_id 10000 --snr 0 
#python 0_addnoise.py mix_speech_fixedSNR --speech_file ./speech.list --noise_file  ./noise.list --workplace ./output_dir --start_id  0 --end_id 10000 --snr 5 
#python 0_addnoise.py mix_speech_fixedSNR --speech_file ./speech.list --noise_file  ./noise.list --workplace ./output_dir --start_id  0 --end_id 10000 --snr 10 


#python 0_addnoise.py mix_speech_ProgressiveSNR --speech_file ./speech.txt --noise_file  ./speech.txt --workplace ./output_dir2 --start_id  0 --end_id 10 --initial_snr -3  --PL_snr_gap 10  --PL_stages_num 2


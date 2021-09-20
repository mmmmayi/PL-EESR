import struct
import torchaudio
import math
import scipy.io.wavfile 
import os
from scipy import signal 
import numpy as np
import pdb 
import argparse
import scipy.io.wavfile as wav_io
import torch
import librosa
import librosa.display
from librosa.filters import mel
import matplotlib.pyplot as plt
import torch.fft
from asvtorch.src.settings.settings import Settings

def stft(x, window, nperseg=400, noverlap=240):
    if len(window)!=nperseg:
        raise ValueError('window length must equal nperseg')
    x=np.array(x) 
    nadd = noverlap - (len (x) -nperseg)%noverlap 
    x =np.concatenate((x, np.zeros(nadd))) 
    step = nperseg - noverlap
    shape=x.shape[:-1] + ((x.shape[-1] - noverlap) //step, nperseg)
    strides=x.strides[:-1] + (step *x.strides[-1], x.strides[-1])
    x=np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides) 
    #print(window)
    x =x* window

    result= np.fft.rfft(x, n=nperseg)
    #print('result:',result)
    return result
    
    
def istft (x, window, nperseg=400, noverlap=240):
    x=np.fft.irfft(x)
    y=np.zeros((len(x)-1 ) * noverlap + nperseg) 
    C1= window[0: 256]
    C2= window[0: 256] + window [256:512]
    C3 =window[256:512]
    y[0:noverlap] =x[0][0:noverlap]/C1
    for i in range(1,len(x)):
        y[i*noverlap: (i+ 1) * noverlap] = (x[i-1][noverlap: nperseg] + x [i][0:noverlap])/ C2 
        y[-noverlap:] = x[len(x) -1][noverlap:] / C3
    return y    
    
def wav2logspec(x, window, nperseg=400, noverlap=240):
    y =stft (x, window, nperseg=nperseg, noverlap=noverlap) 
    return np.log(np.square(abs(y))+1e-8)

def logspec2wav(lps, wave, window, nperseg=512, noverlap=256):
    z = stft (wave, window, nperseg=nperseg, noverlap=noverlap) 
    angle=z/ (np.abs(z) + 1e-8 )
    x=np.sqrt(np.exp(lps))* angle
    x=np.fft.irfft (x)
    y=np.zeros((len(x) -1) * noverlap + nperseg)
    C1= window[0: noverlap]
    C2= window[0: noverlap]  + window[noverlap: nperseg]
    C3= window[noverlap: nperseg]
    y[0: noverlap] =x[0][0: noverlap] / C1 #
    for i in range (1, len(x)):
        y[i*noverlap:(i+1) *noverlap] = (x[i-1][noverlap:nperseg] + x[i][0:noverlap])/C2 
    y[-noverlap:] =x[len(x) -1][noverlap:] /C3  
    return np.int16(y[0: len(wave)])  
    
    
def peak_normalization(wave):
    norm = wave.astype(float)
    norm = norm/ (max(abs(norm))) * (np.exp2(15) -1 )
    return norm.astype(int) 
    
def mix_waveform (input_wav1, input_wav2, snr):
    """
    This function mixes the given two audios with a selected SNR
    """
    print('debug for input:{}'.format(input_wav1))
    try:
        rate, input_data1 = wav_io.read(input_wav1) 
        rate, input_data2 = wav_io.read(input_wav2) 
        input_data1= input_data1.astype('float32') #int16 to float
        input_data2= input_data2.astype('float32')  
        mix_snr= snr

        len1 = len(input_data1)
        len2 = len(input_data2)

        if len2 <= len1:
        #if the noise's length  <speech's length, repeat it
            repeat_data= np.tile(input_data2, int(np.ceil(float(len1)/float(len2))))
            input_data2= repeat_data [0:len1] 
            noise_onset=0 
            noise_offset= len1
        else:
        # randomly select data with equal length
            noise_onset=np.random.randint(low=0, high =len2- len1 , size=1)[0]
            noise_offset =noise_onset +len1
            input_data2 = input_data2[noise_onset: noise_offset]

        assert len(input_data1) ==len(input_data2), 'Two sequence lengths are not equal!!!' 
        scaler = get_scaler(input_data1, input_data2, snr)

        input_data2 /= scaler
        mixed_audio = input_data1+ input_data2

        return mixed_audio, input_data1, input_data2, noise_onset, noise_offset   
    except (ValueError,UnboundLocalError) as e:
        print('Error for {}'.format(e))
        return [0],0,0,0,0
    
    
    
def mix_waveform_given_NoiseInfo (input_wav1, input_wav2, snr, noise_onset, noise_offset):
    """
    This function mixes the given two audios with a selected SNR, which use the same source with the initial SNR.
    """
    rate, input_data1 = wav_io.read(input_wav1) 
    rate, input_data2 = wav_io.read(input_wav2) 
    input_data1= input_data1.astype('float32') #int16 to float
    input_data2= input_data2.astype('float32')  
    mix_snr= snr

    len1 = len(input_data1)
    len2 = len(input_data2)

    if len2 <= len1:
        #if the noise's length  <speech's length, repeat it
        repeat_data= np.tile(input_data2, int(np.ceil(float(len1)/float(len2))) )
        input_data2= repeat_data [0:len1] 
    else:
        input_data2 = input_data2[noise_onset: noise_offset]

    assert len(input_data1) ==len(input_data2), 'Two sequence lengths are not equal!!!' 
    scaler = get_scaler(input_data1, input_data2, snr)

    input_data2 /= scaler
    mixed_audio = input_data1+ input_data2

    return mixed_audio, input_data1, input_data2 
    
    
def get_scaler(speech_data, noise_data, snr):
    """
    Get the scaling factor to match a specific given SNR;
    """
    speech_rms=rms(speech_data)
    noise_rms= rms(noise_data)
    original_rms_ratio = speech_rms/noise_rms
    target_rms_ratio = 10. **(float(snr)/20.)#snr=20 lg(rms(s)/rms(n) 
    scale_factor = target_rms_ratio/ original_rms_ratio 
    return scale_factor

def rms(x):
    """
    first calculate RMS= root mean square
    """
    return np.sqrt(np.mean(np.abs(x) **2, axis=0, keepdims = False) )
    
def save_augment(output_dir,augment_audio):
    wav_io.write(output_dir,16000, np.int16(augment_audio))
 
def save_wav_files(output_dir, speaker_id, mixed_audio, speech_audio, noise_audio):
    mixed_file =os.path.join(output_dir, speaker_id) + '_mix.wav'
    wav_io.write(mixed_file, 16000, np.int16(mixed_audio))
    s_file =os.path.join(output_dir, speaker_id) +'_s.wav'
    wav_io.write(s_file, 16000, np.int16(speech_audio))
    n_file =os.path.join(output_dir, speaker_id) +'_n.wav'
    wav_io.write(n_file, 16000, np.int16(noise_audio))
    if not os.path.getsize(mixed_file) == os.path.getsize(s_file) == os.path.getsize(n_file):
        print ('Error in mixing audio: id-%s in dictionary %s.' %(speaker_id, output_dir) )
    return mixed_file

def extract_lps_fea(mixed_audio, speech_audio, noise_audio):
    mixed_lps = wav2logspec(mixed_audio, window=np.hamming(400), nperseg=400, noverlap=240)
    speech_lps= wav2logspec(speech_audio, window=np.hamming(400), nperseg=400, noverlap=240)
    noise_lps= wav2logspec(noise_audio, window=np.hamming(400), nperseg=400,noverlap=240)
    return mixed_lps, speech_lps, noise_lps 

def _clip_point(frame_len):
    clip_length = 300

    start_point = 0
    end_point = frame_len
    if frame_len>=clip_length:
        start_point = np.random.randint(0, frame_len - clip_length + 1)
        end_point = start_point + clip_length
    return start_point, end_point


def extract(path_list, ids, length=None, usage='train', clip=True):
    features,target_lps = [],[]
    if usage=='train':
        if clip:
            n_frame = torch.min(length)
        
            start_point, end_point = _clip_point(n_frame)
        else:
            end_point = torch.min(length)
            start_point=0
        #print('start:{},end:{}'.format(start_point,end_point))
        for i in range(len(ids)):
            mix_path = ids[i]+'_mix.wav'
            mix_path = os.path.join(path, mix_path)
            mix_lps = generate_lps(mix_path)

            if '_noise' in ids[i]:
                clean_name = ids[i].replace('_noise','')
            elif '_music' in ids[i]:
                clean_name = ids[i].replace('_music','')
            elif '_babble' in ids[i]:
                clean_name = ids[i].replace('_babble','')
            else:
                clean_name = ids[i]
            clean_path = clean_name+'_s.wav'
            clean_path = os.path.join(path, clean_path)
            clean_lps = generate_lps(clean_path)
            
            features.append(mix_lps[:,start_point:end_point])#[201,16662]
            target_lps.append(clean_lps[:,start_point:end_point])
                       
        features = torch.stack(features)
        target_lps = torch.stack(target_lps)
        return features, target_lps
    elif usage=='validation':
        n_frame = torch.min(length)
        for i in range(len(ids)):
            num = ids[i].split('-')[0]
            num = int(num[2:])
            if num>10000:
                idx = (num-10000)//200
                path = os.path.join(path_list[0],str(idx))
            else:
                idx = num//200
                path = os.path.join(path_list[1],str(idx)) 

            mix_path = ids[i]+'_mix.npz'
            mix_path = os.path.join(path, mix_path)
            mix_lps = np.load(mix_path)
            mix_lps = mix_lps['arr_0']
            if '_noise' in ids[i] or '_music' in ids[i] or '_babble' in ids[i]:
                parts = ids[i].split('_')
                del(parts[-1])
                del(parts[-1])
                clean_name = '_'.join(parts)
            elif '_rev' in ids[i]:

                clean_name = ids[i]
            else:
                parts = ids[i].split('_')
                del(parts[-1])
                clean_name = '_'.join(parts)

            clean_path = clean_name+'_s.npz'
            clean_path = np.load(os.path.join(path, clean_path))
            clean_lps = clean_path['arr_0']
            mix_lps = torch.tensor(mix_lps).to(Settings().computing.device)
            clean_lps = torch.tensor(clean_lps).to(Settings().computing.device)

            features.append(mix_lps[:,:n_frame])#[201,1546]
            target_lps.append(clean_lps[:,:n_frame])
        features = torch.stack(features)
        target_lps = torch.stack(target_lps)
       
        return features,target_lps


    else:
        n_frame = length
        for i in range(len(ids)):
            mix_path = ids[i][0]+'_mix.wav'
            mix_path = os.path.join(path, mix_path)
            mix_lps = generate_lps(mix_path)
            features.append(mix_lps[:,:n_frame])
        features = torch.stack(features)
        #print('features:{}'.format(features.shape))#[1,201,598]
        return features
 
def generate_lps(path):
    _, signal = wav_io.read(path)
    x=np.array(signal)
    #nadd = 240 - (len (x) -400)%240
    #x =np.concatenate((x, np.zeros(nadd)))
    #fft_window = np.hamming(400).reshape(400)
    waveform = torch.from_numpy(x)
    waveform = waveform.float()
#    waveform = torch.from_numpy(x)
    pad = 0
    #window = torch.from_numpy(fft_window)
#    window = torch.from_numpy(fft_window)
    n_fft = 400
    hop_length = 160
    win_length = 400
    power = 2
    normalized = False
    center = False
    Spec = torchaudio.transforms.Spectrogram(n_fft =n_fft, win_length=win_length, hop_length=hop_length, power=power, center=center)
    Spec = Spec.to(Settings().computing.device)
    Melscale = torchaudio.transforms.MelScale(30, 16000, 0, 8000, 201)
    Melscale = Melscale.to(Settings().computing.device)
    lps = Spec(waveform.to(Settings().computing.device))
    lps = Melscale(lps)
    return torch.log(lps+1e-8)
 
def read_wav(path):
    sampling_rate, x = wav_io.read(path)
    return sampling_rate, x

def save_wav(audio_np, sr, path):
    wav_io.write(path, sr, np.int16(audio_np))

def wav2lps(x, window=np.hamming(400), nperseg=400, noverlap=240):
    y =stft(x, window, nperseg=nperseg, noverlap=noverlap)
    return np.log(np.square(abs(y))+1e-8)

def lps2mfcc(lps,n_fft=400,sr=16000):
    '''
    lps shape: batch*feat_dim*frame
    '''
    batch,feat,frame = lps.size()
    lps = torch.exp(lps)
    
    mel_basis = torch.from_numpy(mel(sr=sr, n_fft=n_fft, n_mels=30, fmin=0, fmax=sr/2))
    mel_list = torch.randn(batch,mel_basis.shape[0],mel_basis.shape[1])
    mel_list[:,] = mel_basis
    mel_list = mel_list.type_as(lps)
    mel_list.to(Settings().computing.device)
    melspectrogram = torch.bmm(mel_list,lps.transpose(1,2))
#    S = power_to_db(melspectrogram)
    S = torchaudio.functional.amplitude_to_DB(melspectrogram,10.,1e-10,math.log10(max(1e-10, 1.0)),80)
    x_torch = dct_torch(S,'ortho')
#    print('x_torch shape:{}'.format(x_torch.shape))
    mean = x_torch.mean(2).reshape(batch,30,1)
 
    std = x_torch.var(2).reshape(batch,30,1)
    
#    print('mean shape:{},std shape:{}'.format(x_torch.mean(1).shape, x_torch.var(1).shape))
    x_torch = (x_torch-mean)/std
    return x_torch
    
def lps2mfcc_test(lps,n_fft=400,sr=16000):
    lps = torch.exp(lps)
    #print(lps)
    mel_basis = mel(sr=sr, n_fft=n_fft, n_mels=30, fmin=0, fmax=sr/2)
    melspectrogram = torch.from_numpy(np.dot(mel_basis, lps.T))
    #print(melspectrogram)    
    S = power_to_db(melspectrogram)
    #print(S.shape)
  
    x_torch = dct_torch_test(S,'ortho')
    #print('S:',S)    
    #print('x:',x_torch)    
    return x_torch
    '''
    _spectrogrsm = np.e**lps
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=40)
    melspectrogram = np.dot(mel_basis, _spectrogrsm.T)
    S = power_to_db(melspectrogram)
    mfcc = librosa.feature.mfcc(S=S)
    return mfcc
    '''

def power_to_db(S):
    #print(S)
    amin = torch.tensor([1e-10])
    
    ref = torch.abs(torch.tensor([1.0]))
    amin = amin.type_as(S)
    ref = ref.type_as(S)
    top_db = 80.0
    log_spec = 10.0 * torch.log10(torch.max(S, amin))
    
    log_spec -= 10.0 * torch.log10(torch.max(amin, ref))
    
    #print('log_spec size:',log_spec.size())
    #batch_wise_max = log_spec.flatten(1).max(1)[0].unsqueeze(1).unsqueeze(1)
    if S.shape[0]>1:
        
        mid = torch.amax(log_spec,dim=(1,2))-top_db
        mid = mid.reshape(S.shape[0],1,1)
        mid = mid.expand(S.shape[0],S.shape[1],S.shape[2])
        log_spec = torch.max(log_spec, mid) 
    else:
        log_spec = torch.max(log_spec, log_spec.max() - top_db)    
    
    return log_spec

def dct_torch(x,norm):
    x = x.transpose(1,2)
    n = x.shape[-1]
    xx = x[:,...,:n]
    if (torch.remainder(torch.tensor(n),2) == 0):
        xp = 2 * torch.fft.fft(torch.cat( (xx[:,...,::2],torch.flip(xx,[2])[:,...,::2]),dim=2))
    else:
        xp = torch.fft.fft(torch.cat((xx, torch.flip(xx,[2])[:,...,::1]),dim=2))
        xp = xp[:,...,:n]
    w = torch.exp(-1j * torch.arange(n) * math.pi/(2*n))
    w = w.to(Settings().computing.device)
    y = xp*w
    y = y.real
    if norm == 'ortho':
        y[:,:,0]=y[:,:,0]*torch.sqrt(torch.div(1,4*n))
        y[:,:,1:]=y[:,:,1:]*torch.sqrt(torch.div(1,2*n))
    return y.transpose(1,2)

def dct_torch_test(x,norm):
    """
    Discrete Cosine Transform

                      N-1
           y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
                      n=0

    """
    x=x.T
    n = x.shape[-1]
    xx = x[...,:n]
    
    if (torch.remainder(torch.tensor(n),2) == 0):
        #return torch.hstack( (xx[...,::2], torch.flip(xx,[1])[...,::2]) )
        xp = 2 * torch.fft.fft(torch.hstack( (xx[...,::2], torch.flip(xx,[1])[...,::2]) ))
    else:
        #return torch.hstack((xx, torch.flip(xx,[0])[...,::1]))
        xp = torch.fft.fft(torch.hstack((xx, torch.flip(xx,[1])[...,::1])))
        #return xp
        xp = xp[...,:n]
    
    w = torch.exp(-1j * torch.arange(n) * math.pi/(2*n))

    y = xp*w
    y = y.real

    if norm == 'ortho':
        y[:,0]=y[:,0]*torch.sqrt(torch.div(1,4*n))
        y[:,1:]=y[:,1:]*torch.sqrt(torch.div(1,2*n))
    return y.T

 
def lps2irm(lps_noise, lps_clean, window=np.hamming(400), nperseg=400, noverlap=240):
    return np.sqrt( np.exp(lps_clean)/(np.exp(lps_clean) + np.exp(lps_noise) ))
 
 
if __name__=='__main__':
    
    mix_sr, mix_np = read_wav('/data07/mayi/voxceleb_asvtorch/VoxCeleb1/SVModule/id11176-L7ICT0dZny0.wav')
    #mix_np = np.arange(496000)
    mix_np = mix_np.astype('float32')
    mix_mag = wav2lps(mix_np)
    import kaldiio
    mix_mag = kaldiio.load_mat('/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/sitw_dev_librosa/lists/speech_lps.ark:11')
    x_torch_test = lps2mfcc_test(torch.from_numpy(mix_mag),n_fft=400,sr=16000)
    print('1-d:{}'.format(x_torch_test))
    fea = kaldiio.load_mat('/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/sitw_dev_librosa/lists/feature.ark:11')
    print('fea:{}'.format(fea))
    #x_list = []
    #x_list.append(mix_mag)
    #x_list.append(mix_mag)
    #x_multi = torch.from_numpy(np.transpose(np.dstack(x_list),(2,0,1)))
#    print('1D shape:{},2D shape:{}'.format(mix_mag.shape,x_multi.shape))
    #x_torch = lps2mfcc(x_multi,n_fft=400,sr=16000)
    #print('2-d:{}'.format(x_torch))
    
    #print(x_torch.shape)    
    #fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    #librosa.display.specshow(x_torch.numpy(), y_axis='log', x_axis='time',sr=16000)
    #plt.savefig('/data07/mayi/code/asvtorch/asvtorch/src/mask/test.png')

 
 
 

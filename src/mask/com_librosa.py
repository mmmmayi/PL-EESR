import scipy.io.wavfile as wav_io
import librosa.feature
import numpy as np
path = '/data07/mayi/voxceleb_asvtorch/VoxCeleb1/SVModule/id11176-L7ICT0dZny0.wav'
sr,x = wav_io.read(path)
nperseg=400 
noverlap=240
#x=np.arange(496000)
#print(x.shape)
nadd = noverlap - (len (x) -nperseg)%noverlap 
x =np.concatenate((x, np.zeros(nadd)))
S = librosa.feature.mfcc(y=x, sr=16000,n_mfcc=30, n_fft=400, hop_length=160, norm='ortho', win_length=None, window='hamming', center=False, power=2.0, fmin=0,fmax=8000)
#S=librosa.stft(x, n_fft=512, hop_length=256, win_length=None, window='hamming', center=False, dtype=None, pad_mode='reflect')
print(S.shape)


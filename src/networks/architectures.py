
# Copyright 2020 Ville Vestman
#           2020 Kong Aik Lee
# This file is licensed under the MIT license (see LICENSE.txt).
import torchaudio
import torch
import torch.nn as nn
from collections import OrderedDict 
from asvtorch.src.networks.modules import *
from asvtorch.src.settings.settings import Settings
from asvtorch.src.vad.find_threshold import compute_energy, apply_vad, VAD_kaldi, EnergySAD
import asvtorch.src.networks.network_io as network_io
import numpy as np
class BaseNet(nn.Module):
    def __init__(self, feat_dim, n_speakers):
        super().__init__()
        '''
        self.feat_dim_param = torch.nn.Parameter(torch.LongTensor([feat_dim]), requires_grad=False)
        self.n_speakers_param = torch.nn.Parameter(torch.LongTensor([n_speakers]), requires_grad=False)
        self.training_loss = torch.nn.Parameter(torch.Tensor([torch.finfo().max]), requires_grad=False)
        self.consecutive_lr_updates = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)
        '''
        self.feat_dim_param = torch.LongTensor([feat_dim])
        self.n_speakers_param = torch.LongTensor([n_speakers])
        self.training_loss1 = torch.Tensor([torch.finfo().max])
        self.training_loss2 = torch.Tensor([torch.finfo().max])

        self.consecutive_lr_updates = torch.LongTensor([0])
class StandardNetTemplate(BaseNet):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)
        self.feat_dim = feat_dim
        self.n_speakers = n_speakers
        self.dim_featlayer = Settings().network.frame_layer_size
        self.dim_statlayer = Settings().network.stat_size
        self.dim_uttlayer = Settings().network.embedding_size
        self.tdnn_layers = nn.ModuleList()
        self.utterance_layers = nn.ModuleList()
        self.pooling_layer, self.pooling_output_dim = _init_pooling_layer()

    def forward(self, x, forward_mode='train'):
        hidden = []
        for layer in self.tdnn_layers:
            x = layer(x)
            hidden.append(x)

        # To extract "neural features"
        if forward_mode == 'extract_features':
            return x

        # To extract "neural stats" for neural i-vector
        if forward_mode in ('extract_training_stats', 'extract_testing_stats'):
            return self.pooling_layer(x, forward_mode)

        x = self.pooling_layer(x)
        embedding = self.utterance_layers[0].linear(x) 
        hidden.append(embedding)
        if forward_mode == 'extract_embeddings': # Embedding extraction
            return self.utterance_layers[0].linear(x)
            #return self.utterance_layers[0].activation(self.utterance_layers[0].linear(x))

        for layer in self.utterance_layers:
            x = layer(x)

        return x, hidden

class StandardNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)
        self.feat_dim_param = torch.LongTensor([feat_dim])
        self.n_speakers_param = torch.LongTensor([n_speakers])
        self.training_loss = torch.Tensor([torch.finfo().max])
        self.consecutive_lr_updates = torch.LongTensor([0])
        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 2))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 3))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 0))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))


class StandardNet_Mask(nn.Module):
    def __init__(self,feat_dim, n_speakers):
        super().__init__() 
        # other parameters
        self.training_loss = torch.nn.Parameter(torch.Tensor([torch.finfo().max]), requires_grad=False)
        self.consecutive_lr_updates =torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)
        
        #speaker verification module
        self.class_model = StandardNet(30,n_speakers)
        path = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/pretrained/SV_model.pt'
        loaded_states = torch.load(path)
        new_state_dict = loaded_states['model_state_dict']
        '''
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        state_dict = loaded_states['model_state_dict']
        for k, v in state_dict.items():
            namekey = k[7:]
            new_state_dict[namekey]=v
        '''
        self.class_model.load_state_dict(new_state_dict)
        for p in self.class_model.parameters():
            p.requires_grad = False

        # mask module        
        self.mask_model = BLSTM()
        
#        path = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/pretrained/mask_model.pt'
#        loaded_states = torch.load(path)
        
#        new_state_dict = OrderedDict() 
#        for k,v in loaded_states.items():
#            namekey = k.split('.',1)[1]
#            new_state_dict[namekey]=v
#        self.mask_model.load_state_dict(new_state_dict)
        
        # lps2mfcc
        dict = {"win_length":400,
           "hop_length":160,
           "n_fft":400,
           "f_max":8000,
           "pad":0,
           "n_mels":30,
          
           "center":False
           }
        self.Melscale = torchaudio.transforms.MelScale(30, 16000, 0, 8000, 201)
        self.DB=torchaudio.transforms.AmplitudeToDB('power')
        self.DCT = torchaudio.functional.create_dct(n_mfcc=30, n_mels=30, norm='ortho')
        self.DCT = self.DCT.to(Settings().computing.device)

    def _lps2mfcc(self,lps):
        batch = lps.shape[0]
        lps = lps.float()      
        mfcc = lps
        #lps = torch.exp(lps)  
        #mel = self.Melscale(lps)
        #log_mel = self.DB(lps)
        #mfcc = torch.matmul(log_mel.transpose(-2, -1), self.DCT).transpose(-2, -1)
        mean = mfcc.mean(2).reshape(batch,30,1)
        std = mfcc.var(2).reshape(batch,30,1)
        mfcc = (mfcc-mean)/std
        return mfcc 

    def forward(self, x, y, mean,std, forward_mode='train'):
    #def forward(self, x, y,  forward_mode='train'):    
        # noisy speech enhancement
        mix_lps = x.permute(0,2,1)
        #print('mix_lps,',mix_lps.shape)#[64,300,201]
        noisy_irm = self.mask_model(mix_lps)
        #print('irm',noisy_irm.shape) #[64,300,201] 
        #if sum(sum(sum(torch.isnan(noisy_irm))))>0:
            #print('noisy_MFCC:',np.array(noisy_irm[0]))
        estimated_noisy_lps = torch.add(2*torch.log(noisy_irm),mix_lps)
        estimated_noisy_lps = estimated_noisy_lps.permute(0,2,1)
       
        estimated_noisy_lps = estimated_noisy_lps*std+mean
        estimated_noisy_lps.squeeze()
        #print('estimated_noisy_lps',estimated_noisy_lps.shape)#[64,201,300]
        
        #print('noisy_MFCC,',noisy_MFCC.shape)#[64,30,300]
        #if sum(sum(sum(torch.isnan(noisy_irm))))>0:
        #estimated_noisy_lps = x
        if forward_mode=='train':
            noisy_MFCC = self._lps2mfcc(estimated_noisy_lps) 
            # clean speech enhancement
            clean_lps = y.permute(0,2,1)
            clean_irm = self.mask_model(clean_lps)
            estimated_clean_lps = torch.add(2*torch.log(clean_irm),clean_lps)
            estimated_clean_lps = estimated_clean_lps.permute(0,2,1)

            estimated_clean_lps = estimated_clean_lps*std+mean
            estimated_clean_lps.squeeze()

            clean_MFCC = self._lps2mfcc(estimated_clean_lps)


            y = y*std+mean
            target_MFCC = self._lps2mfcc(y) 
     
            noisy_spk, noisy_hidden = self.class_model(noisy_MFCC)
            clean_spk, clean_hidden = self.class_model(clean_MFCC)
            with torch.no_grad():
                _, target_hidden = self.class_model(target_MFCC)

            #if sum(sum(torch.isnan(noisy_embedding)))>0:
                #print(sum(torch.isnan(noisy_embedding)))
                

#                for parameters in self.mask_model.stack_blstm.parameters():
#                   if len(parameters.shape)==2:
#                       print('inf:',sum(sum(torch.isinf(parameters))))
#                       print('nan:',sum(sum(torch.isnan(parameters))))
#                   else:
#                       print('inf:',sum(torch.isinf(parameters)))
#                       print('nan:',sum(torch.isnan(parameters)))
 
            return noisy_spk, clean_spk, noisy_hidden, clean_hidden, target_hidden

        else:
            energy = compute_energy(estimated_noisy_lps)
            
            vad = EnergySAD()
            voice_label = vad.extract(energy)

            #print(voice_label.shape)
            #label_n = voice_label[0].cpu().numpy()
            #print(estimated_noisy_lps)
            #import numpy as no
            #np.set_printoptions(threshold=np.inf)
            #print(np.where(label_n==False)[0])
            #print(len(list(np.where(label_n==False)[0])))
            noisy_lps = apply_vad(voice_label, estimated_noisy_lps)
            #print(noisy_lps.shape)
            #print(noisy_lps)
            #return 
            noisy_MFCC = self._lps2mfcc(noisy_lps)
            
            return estimated_noisy_lps, self.class_model(noisy_MFCC, forward_mode)

class BLSTM(nn.Module):
    def __init__(self):
        super(BLSTM,self).__init__()
        self.input_size = 30
        self.hidden_size = 128
        self.num_layers = 3
        self.output_dim = 30

        self.stack_blstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True, dropout=0.4)
        self.fconn = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, X):
        h0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size).float().cuda()
        c0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size).float().cuda()

        self.stack_blstm.flatten_parameters()
         
        o_, h = self.stack_blstm(X, (h0, c0))
        #if sum(sum(sum(torch.isnan(o_))))>0:
            #for parameters in self.stack_blstm.parameters():
                #print(parameters)
 
        o = self.fconn(o_)

        y_ = self.activation(o)
        
        
        return y_


class StandardSeNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        ser = Settings().network.ser

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_featlayer, 0, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0, ser))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))



class StandardResSeNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        ser = Settings().network.ser

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 0, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0, ser))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))


class LargeResSeNet(StandardNetTemplate):
    def __init__(self, feat_dim, n_speakers):
        super().__init__(feat_dim, n_speakers)

        ser = Settings().network.ser

        self.tdnn_layers.append(CnnLayer(self.feat_dim, self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 4, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 1, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 2, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 3, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 4, ser))
        self.tdnn_layers.append(ResCnnLayer(self.dim_featlayer, 0, ser))
        self.tdnn_layers.append(CnnLayer(self.dim_featlayer, self.dim_statlayer, 0, ser))

        # Pooling layer

        self.utterance_layers.append(LinearBatchNormLayer(self.pooling_output_dim, self.dim_uttlayer))
        #self.utterance_layers.append(LinearReluBatchNormLayer(self.dim_uttlayer, self.dim_uttlayer))
        self.utterance_layers.append(nn.Linear(self.dim_uttlayer, self.n_speakers))



def _init_pooling_layer():
    if Settings().network.pooling_layer_type == 'clustering':
        pooling_layer = ClusteringLayer(Settings().network.stat_size)
        pooling_output_dim = Settings().network.n_clusters * Settings().network.stat_size
    elif Settings().network.pooling_layer_type == 'default':
        pooling_layer = MeanStdPoolingLayer()
        pooling_output_dim = Settings().network.stat_size * 2
    return pooling_layer, pooling_output_dim

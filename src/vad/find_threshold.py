import numpy as np
import kaldiio
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import torch
def compute_energy(features):
    '''
    features: [batch,30,frame]
    energy: [batch,frame]
    '''
    features = torch.exp(features)
    batch = features.shape[0]
    min_ = torch.min(features,2)[0].reshape(batch,30,-1)
    max_ = torch.max(features,2)[0].reshape(batch,30,-1)
    features = (features-min_)/(max_-min_)



    energy = features.sum(dim=1)
    return energy

def apply_vad(label,spectrogram):
    '''
    label: [batch,frame]
    spectrogram: [batch,30,frame]
    new_spec: [batch,30,new_frame]
    '''
    batch = spectrogram.shape[0]
    frame = spectrogram.shape[2]
    n_frame = min(label.sum(dim=1))
    new_spec = []
    for i in range(batch):
        mid=torch.masked_select(spectrogram[i], label[i])
        mid = mid.reshape(30,-1)
        new_spec.append(mid[:,0:n_frame])
    new_spec = torch.stack(new_spec)
    return new_spec

class VAD_kaldi():
    def __init__(self):
        self.vad_energy_threshold = 2
        self.vad_energy_mean_scale = 0.5
        self.vad_frames_context = 5
        self.vad_proportion_threshold = 0.6
        self.max_energy_percentile = 0.9

    def extract(self, energies):
        batch = energies.shape[0]
        frame = energies.shape[1]
        #min_ = torch.min(energies,1)[0].reshape(batch,-1)
        #max_ = torch.max(energies,1)[0].reshape(batch,-1)
        #energies = (energies-min_)/(max_-min_)
        energy_threshold = torch.quantile(energies, self.max_energy_percentile, dim=1)
        energy_threshold = energy_threshold-self.vad_energy_threshold
        result = torch.zeros(batch,frame, dtype=torch.bool)
        for i in range(batch):
            
            #energy_threshold = self.vad_energy_threshold + self.vad_energy_mean_scale * sum(energies[i]) / frame
            
            for t in range(frame):
                num_count = 0
                den_count = 0
                for t2 in range(t-self.vad_frames_context,t+self.vad_frames_context+1):
                    if t2 >= 0 and t2 < frame:
                        den_count += 1
                      
                        if energies[i][t]>energy_threshold[i]:
                            num_count +=1
                if num_count >= den_count * self.vad_proportion_threshold:
                    result[i][t]=True
        return result                
  

class EnergySAD():

    def __init__(self):
        self.name = 'energy_sad'
        self.sad_threshold = 4.2
        self.max_energy_percentile = 0.1

    def initialize(self):
        pass

    def extract(self, energies):
        """ 
        Returns Speech activity labels for the given frames (true if a frame is active speech, false otherwise).
        energy: [batch, frame]
        result: [batch, frame]
        """
        batch = energies.shape[0]
        frame = energies.shape[1]
        #min_ = torch.min(energies,1)[0].reshape(batch,-1)        
        #max_ = torch.max(energies,1)[0].reshape(batch,-1)
        #energies = (energies-min_)/(max_-min_)
        max_energy = torch.quantile(energies, self.max_energy_percentile, dim=1)
        #max_energy = 0.5*energies.mean(dim=1)
        max_energy = max_energy.reshape(batch,1)
        max_energy = max_energy.expand(batch, frame)
        #max_energy = max_energy-self.sad_threshold
        #max_energy = np.max(energies)
        result = torch.gt(energies, max_energy)
        #print(list(np.where(labels==0)))
        #print(len(list(np.where(labels==0)[0])))
        return result

class EnergySAD2():

    def __init__(self):
        self.name = 'energy_sad'
        self.sad_threshold = 2.7
        self.max_energy_percentile = 95

        self.avg_filter_size = 40
        self.avg_filter_threshold = 0.4
        self.min_hole = 30
        self.initialize() 
	

    def initialize(self):
        self.avg_mask = np.ones(self.avg_filter_size) / self.avg_filter_size
        self.dilation_filter = np.ones(self.min_hole)
        self.erosion_filter = np.ones(self.min_hole // 2)


    def extract(self, energies):
        """ 
        Returns Speech activity labels for the given frames (1 if a frame is active speech, 0 otherwise).
            :param frames: Speech frames in rows.
            :param sad_threshold: SAD threshold (typically between 20 and 35).
        """
        #energies = 20 * np.log10(np.std(frames, axis=1) + np.finfo(float).eps)
        #max_energy = np.max(energies)

        max_energy = np.percentile(energies, self.max_energy_percentile)
        print(max_energy)
        labels = energies > max_energy - self.sad_threshold
#        print(labels)
        print(list(np.where(labels==0)))
        print(len(list(np.where(labels==0)[0])))   
        labels = (np.convolve(labels, self.avg_mask, 'same')) > self.avg_filter_threshold
        labels = binary_dilation(labels, structure=self.dilation_filter)
        #labels = (np.convolve(labels, self.avg_mask, 'same')) > self.avg_filter_threshold
        labels = binary_erosion(labels, structure=self.erosion_filter)
        return labels

if __name__=='__main__':
    Energy = EnergySAD()
    scp = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/sitw_dev_librosa/lists/lms_feature.ark:10315945'
    frame = kaldiio.load_mat(scp)
    energy = torch.tensor(frame)
    energy = torch.exp(energy)
    
    #print(energy.shape)
    np.set_printoptions(threshold=np.inf)
    min_ = torch.min(energy,1)[0].reshape(30,-1)
    max_ = torch.max(energy,1)[0].reshape(30,-1)
    energy = (energy-min_)/(max_-min_)
    
#    mean = energy.mean(1).reshape(30,1)
#    std = energy.var(1).reshape(30,1)

#    energy = (energy-mean)/std
    energy = sum(energy)
#    print('min:{},max:{},mean:{}'.format(min(frame),max(frame),frame.mean()))
    #for i in range(100):
        #print(np.argsort(frame)[i])
    energy = energy.unsqueeze(0)
    label = Energy.extract(energy)
    #print(energy.shape)
    label_n = label.numpy()
    #index = np.argwhere(label_n==False)
   

    print(list(np.where(label_n==False)[1]))
    scp = '/data_a8/mayi/test/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/sitw_dev/features/vad_lists.2.ark:11'
    truth = kaldiio.load_mat(scp)
    print(np.where(truth==0)) 

# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

# Main script for VoxCeleb/xvector recipe.

import sys
import os
import subprocess
# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asvtorch', 1)[0])
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import numpy as np

from asvtorch.src.settings.abstract_settings import AbstractSettings
from asvtorch.src.settings.settings import Settings
import asvtorch.recipes.sitw.data_preparation as data_preparation

import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.misc.miscutils import dual_print

#from asvtorch.src.frontend.feature_extractor import FeatureExtractor
from asvtorch.src.utterances.utterance_selector import UtteranceSelector
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.networks.network_testing import extract_embeddings, get_dataloader
import asvtorch.src.networks.network_training_single as network_training
import asvtorch.src.networks.network_io as network_io
from asvtorch.src.networks.training_dataloader_single import get_global_mean_variance
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda
from asvtorch.src.evaluation.scoring import score_trials_plda, prepare_scoring
from asvtorch.src.evaluation.eval_metrics import compute_eer, compute_min_dcf
from asvtorch.src.mask.feature import MFCC,mix_audio_feature
import asvtorch.src.backend.score_normalization as score_normalization
import asvtorch.src.misc.recipeutils as recipeutils
import asvtorch.src.misc.fileutils as fileutils
import argparse
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--step", type=str)
args = parser.parse_args()

def _get_global_mean_variance(path,data,mode='norm_noisy',store=True):

    dataloader = get_dataloader(data)
    clean_mean, standard_clean_dev, noisy_mean, standard_noisy_dev = get_global_mean_variance(dataloader)
    noisy_mean = noisy_mean.reshape(30,1)
    standard_noisy_dev = standard_noisy_dev.reshape(30,1)
    clean_mean = clean_mean.reshape(30,1)
    standard_clean_dev = standard_clean_dev.reshape(30,1)
 
    if store:
        torch.save(noisy_mean, f"{path}/noisy_mean_mel.pt")
        torch.save(standard_noisy_dev, f"{path}/noisy_std_mel.pt")
        torch.save(clean_mean, f"{path}/clean_mean_mel.pt")
        torch.save(standard_clean_dev, f"{path}/clean_std_mel.pt")
 

@dataclass
class RecipeSettings(AbstractSettings):
    start_stage: int = 0
    end_stage: int = 100
    preparation_datasets: Optional[List[str]] = None
    feature_extraction_datasets: List[str] = field(default_factory=lambda: [])
    augmentation_datasets: Dict[str, int] = field(default_factory=lambda: {})
    selected_epoch: int = None  # Find the last epoch automatically

# Initializing settings:
Settings(os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'init_config.py')) 

# Set the configuration file for KALDI MFCCs:
Settings().paths.kaldi_mfcc_conf_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'mfcc.conf')

# Add recipe settings to Settings() (these settings may not be reusable enough to be included in settings.py)
Settings().recipe = RecipeSettings()  

# Get full path of run config file:
run_config_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'run_configs.py') 

# Get run configs from command line arguments
#run_configs = sys.argv[1:]
run_configs = [args.step]
if not run_configs:
    sys.exit('Give one or more run configs as argument(s)!')

# SITW trial lists:
trial_eval_list = [recipeutils.TrialList(trial_list_display_name='SITW eval-core-core', dataset_folder='sitw_eval_librosa', trial_file='sitw_trials_eval_core_core.txt')]

trial_dev_list = [recipeutils.TrialList(trial_list_display_name='SITW dev-core-core', dataset_folder='sitw_dev_librosa', trial_file='sitw_trials_dev_core_core.txt')]
'''
[
SITW core-core=/data_a7/mayi/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs/datasets/sitw/lists/sitw_trials_core_core.txt
]
'''
# Run config loop:
for settings_string in Settings().load_settings(run_config_file, run_configs):

    # Preparation, stage 0
    if Settings().recipe.start_stage <= 0 <= Settings().recipe.end_stage:       
        data_preparation.prepare_datasets(Settings().recipe.preparation_datasets)
        print('finish preparation')

    # Feature extraction, stage 1
    if Settings().recipe.start_stage <= 1 <= Settings().recipe.end_stage:
        #for dataset in Settings().recipe.feature_extraction_datasets:
            #print('dataset:',dataset)
            #FeatureExtractor().extract_features(dataset)
        #print('finish feature extraction')
        #MFCC('sitw_dev_librosa')
#        MFCC('sitw_eval_librosa')
#        MFCC('mask')
        mix_audio_feature('voxceleb1_mask_aug')
        
    # Data augmentation, stage 2
    if Settings().recipe.start_stage <= 2 <= Settings().recipe.end_stage:
        
        for dataset, augmentation_factor in Settings().recipe.augmentation_datasets.items():
            print('dataset:',dataset)
            FeatureExtractor().augment(dataset, augmentation_factor)

    # Network training, stage 5
    if Settings().recipe.start_stage <= 5 <= Settings().recipe.end_stage:
        
        print('Selecting network training data...')
        
        training_data = UtteranceSelector().choose_regex('voxceleb1_mask_aug', '^(?!id11111-G2QZRjUB_VM).*') #training_data is a class of UtteranceList in utterance_list.py
        training_data.combine(UtteranceSelector().choose_all('voxceleb2_mask_aug'))
        print('length of training_data',len(training_data))
        #print('test UtteranceSelector().choose_regex, return list len:{}'.format(len(training_data)))
        
        #label = set(training_data.get_spk_labels())
        training_data.remove_short_utterances(500)  # Remove utts with less than 500 frames
        #label = set(training_data.get_spk_labels())
        training_data.remove_speakers_with_few_utterances(10)  # Remove spks with less than 10 utts
        label = set(training_data.get_spk_labels())
        print('number of speaker:{}'.format(len(label)))
        list_folder = fileutils.get_list_folder('voxceleb2_mask_aug')

        if not os.path.isfile(f"{list_folder}/noisy_mean_mel.pt") \
                or not os.path.isfile(f"{list_folder}/noisy_std_mel.pt"):
            _get_global_mean_variance(list_folder,training_data)         
        mean_train = torch.load(f"{list_folder}/noisy_mean_mel.pt")
        mean_train = mean_train.float()
        std_train = torch.load(f"{list_folder}/noisy_std_mel.pt") 
        std_train = std_train.float()
        mean_lps = torch.load(f"{list_folder}/clean_mean_mel.pt")
        mean_lps = mean_lps.float()
        std_lps = torch.load(f"{list_folder}/clean_std_mel.pt")
        std_lps = std_lps.float()
        print('Selecting PLDA training data...')
        plda_data = UtteranceSelector().choose_regex('voxceleb1_mask_aug', '^(?!id11111-G2QZRjUB_VM).*')
        plda_data.select_random_speakers(500)
        plda_data.name = 'voxceleb1_mask_aug'
        trial_eval = recipeutils.get_trial_utterance_list(trial_eval_list,'sitw_eval_librosa')
        trial_dev = recipeutils.get_trial_utterance_list(trial_dev_list,'sitw_dev_librosa')
        

        result_dev_file = open(fileutils.get_new_results_file(), 'w')

        result_dev_file.write(settings_string + '\n\n')

        eer_stopper = recipeutils.EerStopper()
        training_data.convert_labels_to_numeric()
        #writer = SummaryWriter(comment=str(Settings().network.initial_learning_rate1)+'_'+str(Settings().network.initial_learning_rate2))
        writer = SummaryWriter(comment='JHU'+str(Settings().network.lam)+str(Settings().network.initial_learning_rate1))
        for epoch in range(20, Settings().network.max_epochs, Settings().network.epochs_per_train_call):

            network, stop_flag, epoch = network_training.train_network(training_data, mean_train,std_train, mean_lps, std_lps, writer, epoch)
            #network, stop_flag, epoch = network_training.train_network(training_data,writer,epoch)

            print('finish training')
            path = ['/data_a8/mayi/mix_speech_voxceleb/sitw_eval']
            extract_embeddings(path, trial_eval, network, mean_train, std_train, mean_lps, std_lps, 'extraction')
            path = ['/data_a8/mayi/mix_speech_voxceleb/sitw_dev']
            extract_embeddings(path, trial_dev, network, mean_train, std_train, mean_lps, std_lps, 'extraction')
            path = ['/data_a8/mayi/mix_speech_voxceleb/voxceleb1_fea','/data_a8/mayi/mix_speech_voxceleb/voxceleb2_fea']
            extract_embeddings(path, plda_data, network, mean_train, std_train, mean_lps, std_lps, 'extraction', length=32)
 
            #extract_embeddings(plda_data, network, mean_train, std_train, mean_lps, std_lps)
            network = None
            print('finish extract')

            vector_processor = VectorProcessor.train(plda_data.embeddings, 'cwl', Settings().computing.device)
            trial_eval.embeddings = vector_processor.process(trial_eval.embeddings)
            trial_dev.embeddings = vector_processor.process(trial_dev.embeddings)
            plda_data.embeddings = vector_processor.process(plda_data.embeddings)

            plda = Plda.train_closed_form(plda_data.embeddings, plda_data.get_spk_labels(), Settings().computing.device)

            for trial_list in trial_eval_list:
                trial_file = trial_list.get_path_to_trial_file()
                labels, indices = prepare_scoring(trial_eval, trial_file)
                scores = score_trials_plda(trial_eval, indices, plda)
                eer = compute_eer(scores, labels)[0] * 100
                eer_stopper.add_stopping_eer(eer)
                min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
                output_text = 'EER = {:.4f}  minDCF = {:.4f}  [epoch {}] [{}]'.format(eer, min_dcf, epoch, trial_list.trial_list_display_name)
                dual_print(result_dev_file, output_text)
            dual_print(result_dev_file, '')
            for trial_list in trial_dev_list:
                trial_file = trial_list.get_path_to_trial_file()
                labels, indices = prepare_scoring(trial_dev, trial_file)
                scores = score_trials_plda(trial_dev, indices, plda)
                eer = compute_eer(scores, labels)[0] * 100
                eer_stopper.add_stopping_eer(eer)
                min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
                output_text = 'EER = {:.4f}  minDCF = {:.4f}  [epoch {}] [{}]'.format(eer, min_dcf, epoch, trial_list.trial_list_display_name)
                dual_print(result_dev_file, output_text)            

            dual_print(result_dev_file, '')
            trial_eval.embeddings = None  # Release GPU memory (?)
            trial_dev.embeddings = None
            plda_data.embeddings = None
            torch.cuda.empty_cache()

            #if eer_stopper.stop() or stop_flag:
                #break
             
        result_dev_file.close()

        
        print('done')
        


    # Embedding extraction, stage 7
    if Settings().recipe.start_stage <= 7 <= Settings().recipe.end_stage:
        epoch = Settings().recipe.selected_epoch if Settings().recipe.selected_epoch else recipeutils.find_last_epoch()
#        epoch = 36
        network = network_io.load_network(epoch, Settings().computing.device, 5916)
        #network = network_io.initialize_net(30, 1064)
        #network.to(Settings().computing.device)

        print('Loading trial data...')
        trial_eval_data = recipeutils.get_trial_utterance_list(trial_eval_list,'sitw_eval_librosa')
        trial_dev_data = recipeutils.get_trial_utterance_list(trial_dev_list,'sitw_dev_librosa')


        print('Loading PLDA data...')
        plda_data = UtteranceSelector().choose_regex('voxceleb1_mask_aug', '^(?!id11111-G2QZRjUB_VM).*')
        plda_data.combine(UtteranceSelector().choose_all('voxceleb2_mask_aug')) # use the whole data in testing mode # use the whole data in testing mode
        #plda_data.name = 'voxceleb1_mask_aug'
        list_folder = fileutils.get_list_folder('voxceleb2_mask_aug')
        mean_train = torch.load(f"{list_folder}/noisy_mean_mel.pt")
        std_train = torch.load(f"{list_folder}/noisy_std_mel.pt")
        mean_lps = torch.load(f"{list_folder}/clean_mean_mel.pt")
        std_lps = torch.load(f"{list_folder}/clean_std_mel.pt")

        #print('Extracting trial embeddings...')
        path = ['/data_a8/mayi/mix_speech_voxceleb/sitw_eval']
        extract_embeddings(path, trial_eval_data, network, mean_train, std_train, mean_lps, std_lps, 'extraction')
        trial_eval_data.save('trial_eval_embeddings')
        path = ['/data_a8/mayi/mix_speech_voxceleb/sitw_dev']
        extract_embeddings(path, trial_dev_data, network, mean_train, std_train, mean_lps, std_lps, 'extraction')
        trial_dev_data.save('trial_dev_embeddings')

        #print('Extracting PLDA embeddings...')
#        path = ['/data_a8/mayi/mix_speech_voxceleb/voxceleb1_fea','/data_a8/mayi/mix_speech_voxceleb/voxceleb2_fea']
#        extract_embeddings(path, plda_data, network, mean_train, std_train, mean_lps, std_lps,'extraction',32)
#        plda_data.save('clean_enh_plda_embeddings')

        network = None


    # Embedding processing, PLDA training, Scoring, Score normalization, stage 9
    if Settings().recipe.start_stage <= 9 <= Settings().recipe.end_stage:
        epoch = Settings().recipe.selected_epoch if Settings().recipe.selected_epoch else recipeutils.find_last_epoch()
        trial_eval_data = UtteranceList.load('trial_eval_embeddings')
        trial_dev_data = UtteranceList.load('trial_dev_embeddings')
        plda_data = UtteranceList.load('noisy_enh_plda_embeddings')

        vector_processor = VectorProcessor.train(plda_data.embeddings, 'cwl', Settings().computing.device)
        trial_eval_data.embeddings = vector_processor.process(trial_eval_data.embeddings)
        trial_dev_data.embeddings = vector_processor.process(trial_dev_data.embeddings)
        plda_data.embeddings = vector_processor.process(plda_data.embeddings)

        plda = Plda.train_closed_form(plda_data.embeddings, plda_data.get_spk_labels(), Settings().computing.device)

        # Select score normalization cohort randomly from PLDA training data
        #torch.manual_seed(0)
        #normalization_embeddings = plda_data.embeddings[torch.randperm(plda_data.embeddings.size()[0])[:Settings().backend.score_norm_full_cohort_size], :]

        # Compute s-norm statistics
        #normalization_stats = score_normalization.compute_adaptive_snorm_stats(trial_data.embeddings, normalization_embeddings, plda, Settings().backend.plda_dim, Settings().backend.score_norm_adaptive_cohort_size)

        # Scoring and score normalization
        for trial_list in trial_eval_list:
            trial_file = trial_list.get_path_to_trial_file()
            labels, indices = prepare_scoring(trial_eval_data, trial_file)
            scores = score_trials_plda(trial_eval_data, indices, plda)  
            #scores = score_normalization.apply_snorm(scores, normalization_stats, indices)
            np.savetxt(fileutils.get_score_output_file(trial_list), scores)
            eer = compute_eer(scores, labels)[0] * 100
            min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
            output_text = 'EER = {:.4f}  minDCF = {:.4f}  [epoch {}] [{}]'.format(eer, min_dcf, epoch, trial_list.trial_list_display_name)
            print(output_text)

        for trial_list in trial_dev_list:
            trial_file = trial_list.get_path_to_trial_file()
            labels, indices = prepare_scoring(trial_dev_data, trial_file)
            scores = score_trials_plda(trial_dev_data, indices, plda)  
            #scores = score_normalization.apply_snorm(scores, normalization_stats, indices)
            np.savetxt(fileutils.get_score_output_file(trial_list), scores)
            eer = compute_eer(scores, labels)[0] * 100
            min_dcf = compute_min_dcf(scores, labels, 0.05, 1, 1)[0]
            output_text = 'EER = {:.4f}  minDCF = {:.4f}  [epoch {}] [{}]'.format(eer, min_dcf, epoch, trial_list.trial_list_display_name)
            print(output_text)
            
print('All done!')



 

# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import NamedTuple, Iterable
import os
import sys

from asvtorch.src.settings.settings import Settings
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.utterances.utterance_selector import UtteranceSelector
import asvtorch.src.misc.fileutils as fileutils

def find_last_epoch():
    epoch = 1
    while os.path.exists(os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))):
        epoch += 1
    if epoch == 1:
        sys.exit('ERROR: trying to load model that has not been trained yet ({})'.format(os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))))
    return epoch - 1

class TrialList(NamedTuple):
    trial_list_display_name: str  # Name used for the trial list when printing results
    dataset_folder: str  # Name of the folder where the trial list is (for example 'voxceleb1')
    trial_file: str  # Filename of the trial file (for example 'veri_test.txt')

    def get_path_to_trial_file(self) -> str:
        return fileutils.get_file(self.dataset_folder, self.trial_file)

def get_trial_utterance_list(trial_lists: Iterable[TrialList],name_trail) -> UtteranceList:
    trial_utterances = UtteranceList(name=name_trail)
    for trial_list in trial_lists:
        trial_utterances.combine(UtteranceSelector().choose_from_trials(trial_list.dataset_folder, trial_list.trial_file))
    return trial_utterances


class EerStopper():

    def __init__(self):
        self.best_eer = 100
        self.stopping_eers = []
        self.eer_stop_counter = 0

    def add_stopping_eer(self, eer: float):
        self.stopping_eers.append(eer)

    def stop(self) -> bool:
        stopping_eer = sum(self.stopping_eers) / len(self.stopping_eers)
        self.stopping_eers = []
        print('Stopping EER = {}'.format(stopping_eer))
        if stopping_eer < self.best_eer - Settings().network.minimum_improvement:
            self.best_eer = stopping_eer
            self.eer_stop_counter = 0
        else:
            self.eer_stop_counter += 1
            if self.eer_stop_counter == Settings().network.eer_stop_epochs:
                print('No improvements in {} epochs. Stopping...\n\n'.format(Settings().network.eer_stop_epochs))
                return True
        return False

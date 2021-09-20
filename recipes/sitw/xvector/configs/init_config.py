# pylint: skip-file

computing.network_dataloader_workers = 10
computing.feature_extraction_workers = 44
computing.use_gpu = True
computing.gpu_id = 0

paths.output_folder = '/data07/mayi/code/asvtorch/asvtorch/recipes/sitw/xvector/sitw_outputs'
paths.feature_and_list_folder = 'datasets'  # No need to update this
paths.kaldi_recipe_folder = '/data07/mayi/code/kaldi/egs/voxceleb/v2'
paths.musan_folder = '/data07/mayi/musan'  # Used in Kaldi's augmentation
paths.datasets = {'voxceleb1': '/data07/mayi/voxceleb_asvtorch/VoxCeleb1', 'voxceleb2': '/data07/mayi/voxceleb_asvtorch/VoxCeleb2', 'sitw': '/data07/mayi/sitw_latest', 'noise':'/data07/mayi/musan'}

features.vad_mismatch_tolerance = 0 

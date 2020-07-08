from easydict import EasyDict
import os

cfg = EasyDict()

cfg.train = EasyDict()
cfg.train.batch_size = 128
cfg.train.optim = 'RMSprop'
cfg.train.lr_shed = 'ReduceLROnPlateau'
cfg.train.lr = .0005
cfg.train.log_interval = 1000
cfg.train.nrof_epochs = 35
cfg.train.max_grad_norm = 0.5
cfg.train.patience = 4
cfg.train.ckpt_load = False
cfg.train.l2 = 4e-6
cfg.train.dataset = 'mnli' # ['<x>', '<x>_aug_sampling'], where <x> in ['mnli', 'snli',  'sick']
cfg.train.nrof_classes = 3 if 'mnli' in cfg.train.dataset else 2
cfg.train.shuffle = None # for experiments with examples' order from the paper
cfg.train.replacement_prob = 0.5 # probability of a word being changed while augmenting

cfg.exp = EasyDict()
cfg.exp.project_dir = os.path.split(os.getcwd())[0]
cfg.exp.name = 'cross_test' # ['cross_test', 'augmentation']
cfg.exp.path = os.path.join(cfg.exp.project_dir, 'esim','experiments', cfg.exp.name)
cfg.exp.type = str(cfg.train.batch_size)+'_'+str(cfg.train.lr)[2:]+'_base_testing_new_1w_1c'
cfg.exp.ckpt_path = os.path.join(cfg.exp.path, cfg.train.dataset, 'models_checkpoints', 'checkpoint.pth')
#cfg.exp.logs_dir = os.path.join(cfg.exp.path, cfg.train.dataset, cfg.exp.type, 'logs')

cfg.model = EasyDict()
cfg.model.emb_dim = 300
cfg.model.hid_dim = 300
cfg.model.drput = 0.2
cfg.model.align_vector_len = 3

cfg.eval = EasyDict()
cfg.eval.dataset = 'mnli' # ['mnli', 'snli', 'sick', 'sick_cor', 'rte']
cfg.eval.sick_path = os.path.join(cfg.exp.project_dir,'.data','sick')
cfg.eval.rte_path = os.path.join(cfg.exp.project_dir,'.data','rte')

cfg.dataset = EasyDict()
cfg.dataset.aug_file_path =  os.path.join(cfg.exp.project_dir, 'esim', 'aug_data','overlap_2_words_1_change.pickle')
cfg.dataset.nli_root_dir = os.path.join(cfg.exp.project_dir, '.loc_data')
cfg.dataset.vectors_cache = os.path.join(cfg.exp.project_dir, '.vector_cache')

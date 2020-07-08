from easydict import EasyDict
import os

cfg = EasyDict()

cfg.exp = EasyDict()
cfg.exp.name = 'cross_test' # ['cross_test', 'augmentation']
cfg.exp.path = os.path.join('esim/experiments', cfg.exp.name)
cfg.exp.type = str(cfg.train.batch_size)+'_'+str(cfg.train.lr)[2:]+'_base_testing_new_1w_1c'
cfg.exp.ckpt_path = os.path.join(cfg.exp.path, cfg.train.dataset, 'models_checkpoints', 'checkpoint.pth')
#if not os.path.exists(os.path.dirname(cfg.exp.ckpt_dir)):
 #   os.makedirs(os.path.dirname(cfg.exp.ckpt_dir))
cfg.exp.logs_dir = os.path.join(cfg.exp.path, cfg.train.dataset, cfg.exp.type, 'logs')
if not os.path.exists(cfg.exp.logs_dir):
    os.makedirs(cfg.exp.logs_dir)
#cfg.exp.ckpt_load = False
#cfg.exp.cross_test = False

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

cfg.model = EasyDict()
cfg.model.emb_dim = 300
cfg.model.hid_dim = 300
cfg.model.drput = 0.2
cfg.model.align_vector_len = 3

cfg.eval = EasyDict()
cfg.eval.dataset = 'snli'
cfg.eval.sick_path = '.data/sick'
cfg.eval.rte_path = '.data/rte'

cfg.dataset = EasyDict()
cfg.dataset.aug_file_path = 'esim/aug_data/overlap_2_words_1_change.pickle'
#cfg.dataset.dev_path = './.data/multinli/dev.pkl' #??? change?
#cfg.dataset.test_path = './.data/multinli/test.pkl'
#cfg.dataset.train_path = './.data/multinli/train.pkl'


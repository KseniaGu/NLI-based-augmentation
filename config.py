from easydict import EasyDict
import os

cfg = EasyDict()

cfg.exp = EasyDict()
cfg.exp.name ="cross_test"
cfg.exp_path = os.path.join("./experiments", cfg.exp.name)
#cfg.exp.learn_data_dir = "stats/learn_data"
cfg.exp.ckpt_dir = "models_checkpoints/checkpoint.pth"
#cfg.exp.logs_dir = "./logs"
cfg.exp.ckpt_load = True
cfg.exp.cross_test = True

cfg.train = EasyDict()
cfg.train.batch_size = 128
cfg.train.optim = "RMSprop"
cfg.train.lr_shed = "ReduceLROnPlateau"
cfg.train.lr = .0005
cfg.train.log_interval = 100
cfg.train.nrof_epochs = 40
cfg.train.max_grad_norm = 0.5
cfg.train.patience = 3
cfg.train.ckpt_load = True
cfg.train.l2 = 4e-6
cfg.train.dataset = 'mnli'
cfg.train.nrof_classes = 3

cfg.model = EasyDict()
cfg.model.emb_dim = 300
cfg.model.hid_dim = 300
cfg.model.drput = 0.2
cfg.model.align_vector_len = 3

cfg.eval = EasyDict()
cfg.eval.dataset = 'sick'
cfg.eval.sick_path = './.data/sick/'


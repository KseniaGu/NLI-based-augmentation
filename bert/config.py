from easydict import EasyDict
import torch
import os

cfg = EasyDict()
cfg.exp = EasyDict()
cfg.exp.name = 'base'

cfg.train = EasyDict()
cfg.train.log_interval = 10
cfg.train.ckpt_interval = 1000
cfg.train.dataset = 'rte' # ['<x>', '<x>_aug'], where <x> in ['rte', 'scitail']
cfg.train.model = 'distilbert_base_uncased' # ['distilbert_base_uncased', 'bert_base_uncased']
cfg.train.ckpt_path = os.path.join('bert','model_checkpoints', cfg.train.model.split('_')[0], cfg.exp.name+'_checkpoint')
cfg.train.file_name = 'new_train_comb.tsv' if 'aug' in cfg.train.dataset else 'train.tsv'
cfg.train.file_path = os.path.join('.data', cfg.train.dataset.split('_')[0],cfg.train.file_name)
cfg.train.test_file_name = 'dev.tsv' if 'rte' in cfg.train.dataset else 'test.tsv'
cfg.train.test_file_path = os.path.join('.data', cfg.train.dataset.split('_')[0], cfg.train.test_file_name)
cfg.train.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


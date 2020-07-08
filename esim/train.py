import torch
import torch.nn as nn
import torch.optim as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from tensorboardX import SummaryWriter
import datetime
import random
import time
import os
import pickle
import pandas as pd


from esim.model import Model
from esim.data_loader import DataPrep, EvalDataPrep
from esim.config import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train():
    '''
    A model training class
    '''
    def __init__(self):
        print("Process started: {}".format(datetime.datetime.now()))
        self.dataset = DataPrep(cfg, device)
        if cfg.exp.name == 'cross_test':
            self.eval_dataset = EvalDataPrep(cfg, self.dataset.text, self.dataset.labels, device)
        self.model = Model(self.dataset.get_vocab())
        self.model.to(device)
        print("Parameters count:", sum([param.nelement() for param in self.model.parameters() if param.requires_grad]))

        self.optim = Opt.RMSprop(self.model.parameters(), lr=cfg.train.lr, alpha=0.9)  # , weight_decay = 4e-6)
        self.scheduler = ReduceLROnPlateau(self.optim, patience=cfg.train.patience)
        self.crit = nn.CrossEntropyLoss(reduction='sum')
        self.nrof_epochs = cfg.train.nrof_epochs

        # logging is done in Google Colab notebook
        #self.train_writer = SummaryWriter(os.path.join(cfg.exp.logs_dir, 'train'))
        #self.val_writer = SummaryWriter(os.path.join(cfg.exp.logs_dir, 'val'))
        self.global_step, self.cur_epoch = 0, 0
        self.best_acc, self.best_loss = 0, 1000

        print("Preparation's done: {}".format(datetime.datetime.now()))

    def save_model(self, epoch):
        shuffle = cfg.train.shuffle  if isinstance(cfg.train.shuffle , bool) else True
        params_dict = {"Batch_size": cfg.train.batch_size, 'Shuffle': shuffle, 'Init_LR': cfg.train.lr,
                       'Weight_decay_factor': cfg.train.l2, 'Optim': cfg.train.optim, 'LR_shed': cfg.train.lr_shed}

        if not os.path.exists(os.path.dirname(cfg.exp.ckpt_path)):
            os.makedirs(os.path.dirname(cfg.exp.ckpt_path))

        torch.save({"epoch": epoch,
                    "step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "acc": self.best_acc,
                    "loss": self.best_loss,
                    "params": params_dict},
                   cfg.exp.ckpt_path)

        print("Model saved...")

    def load_model(self):
        try:
            ckpt = torch.load(cfg.exp.ckpt_path)
            self.cur_epoch = ckpt["epoch"] + 1
            self.global_step = ckpt["step"] + 1
            self.model.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.best_acc = ckpt["acc"]
            self.best_loss = ckpt["loss"]

            print("Model loaded...")
        except FileNotFoundError:
            print('There is no checkpoint!')

    def get_random_replacements(self):
        '''
        Makes word replacements for current epoch (for augmentation)
        '''
        nrof_changes = 0
        for i, d in enumerate(self.dataset.train):
            change = self.dataset.elements_changes[i][0]
            if change[2][0] != '_':
                if random.random() > cfg.train.replacement_prob:
                    nrof_changes += 1
                    p_ind, h_ind, words = change
                    old_word = d.premise[p_ind]
                    d.premise[p_ind] = words[0]
                    d.hypothesis[h_ind] = words[0]
                    change[2][0] = old_word

        print("Nrof changes:", nrof_changes)

    def train_epoch(self, epoch):
        self.model.train()

        if 'aug_sampling' in cfg.train.dataset:
            self.get_random_replacements()
            print("Replacement's done")

        self.dataset.tr_iter.init_epoch()
        cur_loss, nrof_cor_predicts,nrof_predicts = 0.0, 0,0

        for batch_idx, batch in enumerate(self.dataset.tr_iter):
            self.optim.zero_grad()
            try:
                predict = self.model(batch)
                linear3_params = torch.cat([x.view(-1) for x in self.model.projection3.parameters()])
                self.linear3_regular = cfg.train.l2 * torch.norm(linear3_params, 2)
                loss = self.crit(predict, batch.label) + self.linear3_regular
                loss.backward()

                cur_loss += loss.item()
                nrof_cor_predicts += (torch.max(predict, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                nrof_predicts += batch.batch_size

                self.optim.step()
                self.global_step += 1

            except RuntimeError as excp:
                if "out of memory" in str(excp):
                    print("Out of memory!")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise excp

            if batch_idx % cfg.train.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tTrain loss: {:.6f}\tTrain accuracy: {}'.format(
                    epoch, batch_idx * batch.batch_size, len(self.dataset),
                           cur_loss / nrof_predicts, 100. * nrof_cor_predicts / nrof_predicts))
                cur_loss, nrof_cor_predicts, nrof_predicts = 0.0, 0, 0


    def validate(self):
        self.model.eval()
        self.dataset.dev_iter.init_epoch()
        nrof_cor_predicts, nrof_predicts, cur_loss = 0, 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataset.dev_iter):
                predict = self.model(batch)
                loss = self.crit(predict, batch.label)

                nrof_cor_predicts += (torch.max(predict, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                nrof_predicts += batch.batch_size
                cur_loss += loss.item()

            val_loss = cur_loss / nrof_predicts
            val_acc = 100. * nrof_cor_predicts / nrof_predicts
            return val_loss, val_acc

    def train(self):
        if cfg.exp.ckpt_load:
            self.load_model()

        for epoch in range(self.cur_epoch, self.nrof_epochs + 1):
            t = time.time()
            self.train_epoch(epoch)
            print("Epoch trained")
            val_loss, val_acc = self.validate()
            print("Validation done")
            t = time.time() - t
            if val_loss < self.best_loss:
                self.best_loss, self.best_acc  = val_loss, val_acc

            self.scheduler.step(val_loss)

            print(
                "Time taken: {:.3f}, epoch: {}, validation loss: {:.6f}, validation accuracy: {:.6f} ".format(t, epoch,
                                                                                                              val_loss,
                                                                                                              val_acc))
            print("Lr state", self.optim.state_dict()["param_groups"][0]["lr"])

            self.save_model(epoch)

    def evaluate(self):
        if cfg.exp.ckpt_load:
            self.load_model()
        self.model.eval()

        data_iter = self.eval_dataset.dev_iter
        data_iter.init_epoch()
        nrof_cor_predicts, nrof_predicts, cur_loss = 0, 0, 0


        with torch.no_grad():
            for batch_idx, batch in enumerate(data_iter):
                predict = self.model(batch)
                loss = self.crit(predict, batch.label)

                p_label = torch.max(predict, 1)[1].view(batch.label.size())

                if cfg.eval.dataset == 'rte' and cfg.train.dataset=='snli':
                    p_label = torch.where(p_label!=2, p_label, torch.ones(p_label.shape,dtype=torch.int64))
                elif cfg.eval.dataset == 'rte' and cfg.train.dataset=='mnli':
                    p_label = torch.where( p_label!=1, p_label, torch.zeros(p_label.shape, dtype=torch.int64))

                nrof_cor_predicts += (p_label  == batch.label).sum().item()
                nrof_predicts += batch.batch_size
                cur_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(batch_idx)

        val_loss = cur_loss / nrof_predicts
        val_acc = 100. * nrof_cor_predicts / nrof_predicts

        data_dict = {"results": {"val loss": val_loss, "val acc": val_acc},
                     "params": {"train dataset": cfg.train.dataset,
                               "eval dataset": cfg.eval.dataset,
                                "model": "base"}}

        path_to_pkl = os.path.join(cfg.exp_path, cfg.train.dataset + '_train', 'results', 'cross_data_' + cfg.eval.dataset)
        print(data_dict)

        with open(path_to_pkl, "wb") as f:
            pickle.dump(data_dict, f)

        return val_loss, val_acc


def check_results():
    sets = ["snli", "mnli", "sick", "sick_cor", "rte"]
    tr_sets = sets[:2]
    res = dict(zip(sets, [[] for _ in range(len(sets))]))
    for p1 in tr_sets:
        for p2 in sets:
            path = os.path.join(cfg.exp.path, p1 + '_train', 'results', 'cross_data_' + p2)
            with open(path, "rb") as f:
                data = pickle.load(f)
                res[data['params']['eval dataset']].append(data['results']['val acc'])

    data = pd.DataFrame(res, index=tr_sets)
    print(data)





#Tr_model = Train()
#Tr_model.train()
#Tr_model.evaluate()

check_results()

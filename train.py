import torch
import torch.nn as nn
import torch.optim as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import time
import os
import pickle
import pandas as pd
from io import StringIO

from model import Model
from data_loader import DataPrep, EvalDataPrep
from config import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train():
    '''
    A model training class.
    '''
    def __init__(self):
        print("Process started: {}".format(datetime.datetime.now()))
        self.dataset = DataPrep(cfg, device)
        if cfg.exp.cross_test:
            self.eval_dataset = EvalDataPrep(cfg, self.dataset.text, self.dataset.labels, device)
        self.model = Model(self.dataset.get_vocab())
        print("Parameters count:", sum([param.nelement() for param in self.model.parameters() if param.requires_grad]))
        self.model.to(device)

        self.optim = Opt.RMSprop(self.model.parameters(), lr=cfg.train.lr, alpha=0.9)  # , weight_decay = 4e-6)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 5, gamma = 0.1_cross_test)
        self.scheduler = ReduceLROnPlateau(self.optim, patience=cfg.train.patience)
        self.crit = nn.CrossEntropyLoss(reduction='sum')
        self.nrof_epochs = cfg.train.nrof_epochs

        #self.patience_countr = 0
        self.global_step = 0
        self.cur_epoch = 0


        print("Preparation's done: {}".format(datetime.datetime.now()))

    def save_model(self, epoch):
        save_path = os.path.join(cfg.exp_path, cfg.train.dataset + '_train', cfg.exp.ckpt_dir)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save({"epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.scheduler.state_dict()},
                   save_path)

        print("Model saved...")

    def load_model(self):
        load_path = os.path.join(cfg.exp_path, cfg.train.dataset + '_train', cfg.exp.ckpt_dir)
        ckpt = torch.load(load_path, map_location = device)
        self.cur_epoch = ckpt["epoch"] + 1
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        print("Model loaded...")

    def train_epoch(self, epoch):
        '''
        Training with L1 regularization applied to FF layers
        '''
        self.model.train()
        self.dataset.tr_iter.init_epoch()
        nrof_cor_predicts, nrof_predicts, cur_loss = 0, 0, 0

        for batch_idx, batch in enumerate(self.dataset.tr_iter):
            self.optim.zero_grad()
            try:
                predict = self.model(batch)
                linear3_params = torch.cat([x.view(-1) for x in self.model.projection3.parameters()])
                self.linear3_regular = cfg.train.l2 * torch.norm(linear3_params, 2)
                loss = self.crit(predict, batch.label) + self.linear3_regular

                nrof_cor_predicts += (torch.max(predict, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                nrof_predicts += batch.batch_size
                cur_loss += loss.item()

                loss.backward()
            except RuntimeError as excp:
                if "out of memory" in str(excp):
                    print("Out of memory!")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise excp

            self.optim.step()
            if batch_idx % cfg.train.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch), len(self.dataset.train),
                           loss.item() / len(batch)))

        tr_loss = cur_loss / nrof_predicts
        tr_acc = 100. * nrof_cor_predicts / nrof_predicts
        return tr_loss, tr_acc

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
            tr_loss, tr_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            t = time.time() - t

            self.scheduler.step(val_loss)

            print(
                "Time taken: {:.3f}, epoch: {}, train loss: {:.6f}, train accuracy: {:.6f}, validation loss: {:.6f}, validation accuracy: {:.6f} ".format(
                    t, epoch, tr_loss, tr_acc, val_loss, val_acc))
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

                nrof_cor_predicts += (torch.max(predict, 1)[1].view(batch.label.size()) == batch.label).sum().item()
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

        path_to_pkl = os.path.join(cfg.exp_path, 'results', 'cross_data_' + cfg.train.dataset + '_' + cfg.eval.dataset)
        #path_to_pkl = cfg.exp_path + "/cross_data_" + cfg.train.dataset + '_' + cfg.eval.dataset
        #print(path_to_pkl)

        with open(path_to_pkl, "wb") as f:
            pickle.dump(data_dict, f)

        return val_loss, val_acc


def check_results():
    sets = ["snli", "mnli", "sick"]
    res =  dict(zip(["snli", "mnli", "sick"],[[],[],[]]))
    for p1 in sets[:-1]:
        for p2 in sets:
            path = os.path.join(cfg.exp_path, p1 + '_train', 'results', 'cross_data_' + p2)
            with open(path, "rb") as f:
                data = pickle.load(f)
                res[data['params']['eval dataset']].append(data['results']['val acc'])

    data = pd.DataFrame(res, index=sets[:-1])
    print(data)





#Tr_model = Train()
#Tr_model.evaluate()

check_results()

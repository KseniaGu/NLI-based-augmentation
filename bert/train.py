from transformers import BertForSequenceClassification, AdamW, \
    DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from bert.config import cfg
from bert.data_loader import DataPrep
import torch

import datetime
import time

'''
Here is incomplete version of code, as everything was done in Google Colab and adjusted for it 
'''

def format_time(elapsed):
    '''
    Takes time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class Train():
    '''
    A model training class
    '''
    def __init__(self):
        if cfg.train.model == 'bert_base_uncased':
            print('bert')
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            print('dbert')
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False,
            )
        self.training_stats = []
        self.global_step = 0
        self.total_steps = 300
        if cfg.train.device == torch.device("cuda"):
            self.model.cuda()

    def validate(self, validation_dataloader):
        t1 = time.time()
        self.model.eval()
        total_eval_loss, nrof_cor_predicts, nrof_samples = 0., 0, 0

        for batch in validation_dataloader:
            with torch.no_grad():
                b_input_ids = batch[0].to(cfg.train.device)
                b_input_mask = batch[1].to(cfg.train.device)
                b_labels = batch[2].to(cfg.train.device)

                loss, logits = self.model(b_input_ids,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

                total_eval_loss += loss.item()
                nrof_cor_predicts += (torch.max(logits, 1)[1].view(b_labels.size()) == b_labels).sum().item()
                nrof_samples += len(b_labels)

        avg_val_accuracy = nrof_cor_predicts / nrof_samples
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = (time.time() - t1)
        print("  Validation took: {:}".format(validation_time))

        return avg_val_accuracy, avg_val_loss


    def train(self, train_dataloader, validation_dataloader):
        optimizer = AdamW(self.model.parameters(),
                              lr=2e-5,
                              eps=1e-8,
                              weight_decay=0.0001,
                              )

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=self.total_steps)

        t0 = time.time()
        cur_loss, entail_loss, not_entail_loss = 0.0, 0.0, 0.0

        while self.global_step < self.total_steps:
            for step, batch in enumerate(train_dataloader):
                self.model.train()

                b_input_ids = batch[0].to(cfg.train.device)
                b_input_mask = batch[1].to(cfg.train.device)
                b_labels = batch[2].to(cfg.train.device)
                # separate losses for both labels
                not_ent_inds = torch.nonzero(b_labels, as_tuple=True)
                ent_inds = torch.nonzero(b_labels == 0, as_tuple=True)
                self.model.zero_grad()

                ent_loss, ent_logits = self.model(b_input_ids[ent_inds],
                                                  attention_mask=b_input_mask[ent_inds],
                                                  labels=b_labels[ent_inds])
                not_ent_loss, not_ent_logits = self.model(b_input_ids[not_ent_inds],
                                                          attention_mask=b_input_mask[not_ent_inds],
                                                          labels=b_labels[not_ent_inds])

                entail_loss += ent_loss.item()
                not_entail_loss += not_ent_loss.item()
                loss = ent_loss + not_ent_loss
                cur_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if self.global_step % cfg.train.log_interval == 0 and not self.global_step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(self.global_step,
                                                                                len(train_dataloader), elapsed))
                    val_acc, val_loss = self.validate(validation_dataloader)

                    self.training_stats.append(
                        {
                            'iter': self.global_step,
                            'Training Loss': cur_loss / cfg.train.log_interval,
                            'Training ent loss': entail_loss / cfg.train.log_interval,
                            'Training not ent loss': not_entail_loss / cfg.train.log_interval,
                            'Valid. Loss': val_loss,
                            'Valid. Accur.': val_acc
                        })

                    print("tr loss: {}\ntr_ent_loss: {}\ntr_not_ent_loss:{}\nval loss: {}\nval acc:{}".format(
                        cur_loss / cfg.train.log_interval, entail_loss / cfg.train.log_interval,
                        not_entail_loss / cfg.train.log_interval, val_loss, val_acc))
                    cur_loss, entail_loss, not_entail_loss = 0.0, 0.0, 0.0

                if self.global_step % cfg.train.ckpt_interval == 0 and not self.global_step == 0:
                    torch.save({"model": self.model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                },
                               cfg.train.ckpt_path)
                    print("Model saved")

                self.global_step += 1
                if self.global_step >= self.total_steps:
                    break

        return self.training_stats


T = Train()
DP = DataPrep()
training_stats = T.train(DP.tr_loader, DP.tst_loader)
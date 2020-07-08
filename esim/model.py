import torch
import torch.nn as nn
from esim.config import cfg


class PackedLSTM(nn.Module):
    '''
    Packing tensor a and passing it through lstm
    '''
    def __init__(self, hid_dim=300, batch_first=True, bidirectional=True):
        super(PackedLSTM, self).__init__()
        self.lstm = nn.LSTM(hid_dim, hid_dim, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, a):
        a_seq_lens = [x.size()[0] for x in a]
        packed_a = nn.utils.rnn.pack_padded_sequence(a, a_seq_lens, batch_first=True, enforce_sorted=False)
        enc_a, _ = self.lstm(packed_a)
        enc_a, _ = nn.utils.rnn.pad_packed_sequence(enc_a, batch_first=True)
        return enc_a

class AlignMatr(nn.Module):
    '''
    A matrix of scores between the elements of sequences in tensors a and b
    '''
    def __init__(self):
        super(AlignMatr, self).__init__()

    def forward(self, a, b):
        e = torch.bmm(a, b.permute(0, 2, 1))
        return e


class AlignVector(nn.Module):
    '''
    A vector representing important parts of sent
        for the second sentence in a pair
    Args:
        sent: a tensor of sequences, which importances are counted
        e: an alignment matrix (AlignMatr result)
        transp_need: True, when input is premise
    '''
    def __init__(self):
        super(AlignVector, self).__init__()

    def _permute_dim(self, sent, new_dim_size):
        sent = sent.unsqueeze(3)
        sent = sent.repeat(1, 1, 1, new_dim_size)
        return sent.permute(0, 3, 1, 2)

    def forward(self, sent, e, transp_need=False):
        if transp_need:
            e = e.permute(0, 2, 1)
        e = torch.exp(e - e.max(2, keepdim=True)[0])
        e_sum = torch.sum(e, 2, keepdim=True)
        es = e / e_sum
        new_sent = self._permute_dim(sent, es.size()[1])
        es_sent = es.unsqueeze(3) * new_sent
        sent_sum = torch.sum(es_sent, 2)
        return sent_sum


class Model(nn.Module):
    '''
    Implementation of the ESIM-based model
    Args:
        vocab: vocabulary, built from training dataset
        bidirectional: True if BiLSTM is used
        agr_type: "lstm" for aggregating word vectors via lstm last state
				  "bilstm" for aggregating word vectors via bilstm last states concatenation
                  "sum" for aggregating via summarizing word vectors
    '''
    def __init__(self, vocab=None, agr_type="lstm"): 
        super(Model, self).__init__()
        lstm_dirs_count = 2 if agr_type=="bilstm" else 1
        self.embedding = nn.Embedding(len(vocab), cfg.model.emb_dim).from_pretrained(vocab.vectors)  # freeze = True
        self.projection1 = nn.Linear(cfg.model.emb_dim, cfg.model.hid_dim)
        self.dropout = nn.Dropout(p=cfg.model.drput)
        self.lstm = nn.PackedLSTM(cfg.model.hid_dim, cfg.model.hid_dim, batch_first=True, bidirectional=(lstm_dirs_count==2))
        self.relu = nn.ReLU()
        self.alignment = AlignMatr()
        self.align_vector = AlignVector()
        self.projection2 = nn.Linear(lstm_dirs_count * cfg.model.align_vector_len * cfg.model.hid_dim, cfg.model.hid_dim)
        self.projection3 = nn.Linear(lstm_dirs_count * 2 * cfg.model.hid_dim, lstm_dirs_count * 2 * cfg.model.hid_dim)
        # self.batch_norm = nn.BatchNorm1d(lstm_dirs_count*2*cfg.model.hid_dim) # doesn't bring any gain 
        self.projection_out = nn.Linear(lstm_dirs_count * 2 * cfg.model.hid_dim, cfg.train.nrof_classes)
        self.sftmx = nn.Softmax()
        self.agr_type = agr_type

    def forward(self, batch):
        '''
        Args:
            batch: A batch of varaible length sequences of word indices
                representing premise and hypothesis
        '''
        emb_prem = self.embedding(batch.premise)
        emb_hyp = self.embedding(batch.hypothesis)
        proj_prem = self.relu(self.projection1(emb_prem))
        proj_hyp = self.relu(self.projection1(emb_hyp))
        enc_prem, _ = self.lstm(proj_prem)
        enc_hyp, _ = self.lstm(proj_hyp)
        # projection needed? F(a) , F(b) before alignment
        align_e = self.alignment(enc_prem, enc_hyp)
        align_prem = self.align_vector(enc_prem, align_e, transp_need=True)
        align_hyp = self.align_vector(enc_hyp, align_e)
        conc_prem = torch.cat((enc_prem, align_hyp, enc_prem - align_hyp), 2)
        conc_hyp = torch.cat((enc_hyp, align_prem, enc_hyp - align_prem), 2)
        proj_prem = self.relu(self.projection2(conc_prem))
        proj_hyp = self.relu(self.projection2(conc_hyp))
        if 'lstm' in self.agr_type:
            _, (agr_prem, _) = self.lstm(proj_prem)
            agr_prem = torch.cat((agr_prem[0], agr_prem[1]), 1)
            _, (agr_hyp, _) = self.lstm(proj_hyp)
            agr_hyp = torch.cat((agr_hyp[0], agr_hyp[1]), 1)
        else:
            agr_prem = proj_prem.sum(dim=1)
            agr_hyp = proj_hyp.sum(dim=1)
        prem_hyp = torch.cat((agr_prem, agr_hyp), 1)
        prem_hyp = self.dropout(prem_hyp)
        enc_prem_hyp = self.projection3(prem_hyp)
        prem_hyp = self.dropout(self.relu(enc_prem_hyp))
        # norm_prem_hyp = self.batch_norm(prem_hyp)
        # enc_prem_hyp = self.projection3(norm_prem_hyp)
        enc_prem_hyp = self.projection3(prem_hyp)
        prem_hyp = self.dropout(self.relu(enc_prem_hyp))
        enc_prem_hyp = self.projection_out(prem_hyp)
        predictions = self.sftmx(enc_prem_hyp)

        return predictions

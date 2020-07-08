from bert.config import cfg
import pandas as pd
from transformers import DistilBertTokenizer, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

class DataPrep:
    def __init__(self):
        tr_sents, tst_sents, tr_labs, tst_labs = self._read_data()
        (tr_ids, tr_mask, tr_labs), (tst_ids, tst_mask, tst_labs) = \
                self._get_tokens(tr_sents, tr_labs), self._get_tokens(tst_sents, tst_labs)
        self.tr_loader = self._get_dataloader(tr_ids, tr_mask, tr_labs)
        self.tst_loader = self._get_dataloader(tst_ids, tst_mask, tst_labs, phase='test')

    def _read_data(self):
        '''
        Reads dataset
        '''
        if not 'rte' in cfg.train.dataset:
            f_names = ['premise', 'hypothesis', 'label']
        else:
            f_names = ['index', 'sentence1', 'sentence2', 'label']

        tr_df = pd.read_csv(cfg.train.file_path, delimiter="\t", header=None, names=f_names)
        tst_df = pd.read_csv(cfg.train.test_file_path, delimiter="\t", header=None, names=f_names)
        print('Number of training sentences: {:,}\n'.format(tr_df.shape[0]))
        print('Number of training sentences: {:,}\n'.format(tst_df.shape[0]))

        start_ind = 1 if 'rte' in cfg.train.dataset else 0
        if not 'rte' in cfg.train.dataset:
            tr_premises = tr_df.premise.values[start_ind:]
            tr_hypothesises = tr_df.hypothesis.values[start_ind:]
            tst_premises = tst_df.premise.values[start_ind:]
            tst_hypothesises = tst_df.hypothesis.values[start_ind:]
        else:
            tr_premises = tr_df.sentence1.values[start_ind:]
            tr_hypothesises = tr_df.sentence2.values[start_ind:]
            tst_premises = tst_df.sentence1.values[start_ind:]
            tst_hypothesises = tst_df.sentence2.values[start_ind:]

        tr_sentences = list(zip(tr_premises, tr_hypothesises))
        tst_sentences = list(zip(tst_premises, tst_hypothesises))
        tr_labels = tr_df.label.values[start_ind:]
        tst_labels = tst_df.label.values[start_ind:]

        if 'rte' in cfg.train.dataset:
            label_map = lambda label: 1 if label == 'not_entailment' else 0
        else:
            label_map = lambda label: 0 if label == 'entails' else 1

        tr_labels = [label_map(x) for x in tr_labels]
        tst_labels = [label_map(x) for x in tst_labels]

        return tr_sentences, tst_sentences, tr_labels, tst_labels

    def _get_tokens(self, sentences, labels):
        '''
        Tokenization and numericalization
        Args:
            sentences (list of tuples): tuples of sentences (premise, hypothesis)
            labels (list): numericalized labels
        '''
        if cfg.train.model == 'bert_base_uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        input_ids, attention_masks = [], []

        for i, prem_hyp in enumerate(sentences):
            prem, hyp = prem_hyp
            encoded_dict = tokenizer.encode_plus(
                prem,
                hyp,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        print('Original: ', sentences[0])
        print('Token IDs:', input_ids[0])

        return input_ids, attention_masks, labels

    def _get_dataloader(self, input_ids, attention_masks, labels, batch_size=32, phase='train'):
        '''
        Builds dataloader
        Args:
            input_ids (tensor): token IDs
            attention_masks (tensor): masks covering sentence tokens
            labels (tensor): numericalized labels
        '''
        dataset = TensorDataset(input_ids, attention_masks, labels)

        if phase=='train':
            dataloader = DataLoader(
                        dataset,
                        sampler = RandomSampler(dataset),
                        batch_size = batch_size
                    )
        else:
            dataloader = DataLoader(
                        dataset,
                        batch_size = 128
                    )

        return dataloader







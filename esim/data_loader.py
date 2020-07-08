import torch
from torchtext import data, datasets
from torchtext.vocab import GloVe
import pickle
# from allennlp.modules.elmo import batch_to_ids # for changing static vector representations to contextualized
from esim.config import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataPrep():
    '''
    Prepares a dataset for training: loads dataset (MNLI or SNLI),
        splits it into sets, makes vocab and iterator
    '''
    def __init__(self, cfg, device):
        sort_key = lambda ex: data.interleave_keys(len(ex.premise), len(ex.hypothesis))
        self.text = data.Field(lower=True, tokenize='spacy', batch_first=True)
        self.labels = data.Field(lower=True, sequential=False, is_target=True, unk_token=None) # preprocessing=self._prepare_label()
        self.new_words = []

        if cfg.train.dataset == 'mnli':
            self.train, self.dev, self.test = datasets.MultiNLI.splits(self.text, self.labels)

        if cfg.train.dataset == 'snli':
            self.train, self.dev, self.test = datasets.SNLI.splits(self.text, self.labels)

        if cfg.train.dataset == 'mnli_aug_sampling':
            self.train, self.dev, self.test = datasets.MultiNLI.splits(self.text, self.labels)

            with open(cfg.dataset.aug_file_path, 'rb') as f:
                self.elements_changes = pickle.load(f)
                for i,ind in enumerate(self.elements_changes):
                    for change in ind:
                        if isinstance(change[2], list):
                            if change[2][0]!='_':
                                self.new_words.append(change[2][0])
                        else:
                            print('Not list! In element:', i)

        self.text.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name="840B", dim=300))
        self.labels.build_vocab(self.train)

        if not cfg.exp.name == 'cross_test':
            self.tr_iter, self.dev_iter, self.tst_iter = \
                data.BucketIterator.splits((self.train, self.dev, self.test),
                                           batch_sizes=[cfg.train.batch_size] * 3,
                                           device=device,sort_key=sort_key)

    def _prepare_label(self):
        conv_func = lambda label: 'contradiction' if label=='not_entailment' else label
        return data.Pipeline(convert_token=conv_func)

    def get_vocab(self):
        return self.text.vocab

    def __len__(self):
        return len(self.train)

class EvalDataPrep():
    '''
    Prepares a dataset for evaluating: loads dataset (MNLI, SNLI or SICK),
        splits it into sets. It's used for cross-dataset testing
        Sick dataset is found in https://github.com/felipessalvatore/NLI_datasets
    '''
    def __init__(self, cfg, text_field, label_field, device):
        if cfg.eval.dataset == 'mnli':
            self.train, self.dev, self.test = datasets.MultiNLI.splits(text_field, label_field)

        if cfg.eval.dataset == 'snli':
            self.train, self.dev, self.test = datasets.SNLI.splits(text_field, label_field)

        if cfg.eval.dataset == 'sick':
            self.train, self.dev, self.test = data.TabularDataset.splits(
                path=cfg.eval.sick_path,
                train='train_dev.csv',
                validation='train_dev.csv',
                test='train_dev.csv',
                format='csv',
                fields=[('premise', text_field),
                        ('hypothesis', text_field),
                        ('label', label_field)
                        ],
                skip_header=True)

        if cfg.eval.dataset == 'sick_cor':
            self.train, self.dev, self.test = data.TabularDataset.splits(
                path=cfg.eval.sick_path,
                train='SICK_corrected.tsv',
                validation='SICK_corrected.tsv',
                test='SICK_corrected.tsv',
                format='tsv',
                fields= {'sentence_A': ('premise', text_field),
                 'sentence_B': ('hypothesis', text_field),
                 'entailment_label': ('label', label_field)},
                skip_header=False)

        if cfg.eval.dataset == 'rte':
            self.train, self.dev, self.test = data.TabularDataset.splits(
                path=cfg.eval.rte_path,
                train='train_dev.csv',
                validation='train_dev.csv',
                test='train_dev.csv',
                format='csv',
                fields=[('premise', text_field),
                        ('hypothesis', text_field),
                        ('label', label_field)
                        ],
                skip_header=True)

        sort_key = lambda ex: data.interleave_keys(len(ex.premise), len(ex.hypothesis))
        self.tr_iter, self.dev_iter, self.tst_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[cfg.train.batch_size] * 3,
                                       device=device, sort_key=sort_key)

    def show_statistics(self):
        classes_distrib =[0,0,0]

        for i, data in enumerate(self.train):
            if cfg.eval.dataset == 'rte':
                if data.label == 'entailment':
                    classes_distrib[0] += 1
                else:
                    classes_distrib[2] += 1
            else:
                if data.label == 'entailment':
                    classes_distrib[0] += 1
                elif data.label == 'neutral':
                    classes_distrib[1] += 1
                else:
                    classes_distrib[2] += 1

        print(classes_distrib)





if __name__ == '__main__':
    DP = DataPrep(cfg, device)
    EDP = EvalDataPrep(cfg, DP.text, DP.labels, device)
    EDP.show_statistics()





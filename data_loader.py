from torchtext import data, datasets
from torchtext.vocab import GloVe

class DataPrep():
    '''
    Preparing a dataset for training: loads dataset (MNLI or SNLI),
        splits it into sets, makes vocab and iterator
    '''
    def __init__(self, cfg, device):
        self.text = data.Field(lower=True, tokenize='spacy', batch_first=True)
        self.labels = data.Field(sequential=False, is_target=True, unk_token=None)
        if cfg.train.dataset == 'mnli':
            self.train, self.dev, self.test = datasets.MultiNLI.splits(self.text, self.labels)
        if cfg.train.dataset == 'snli':
            self.train, self.dev, self.test = datasets.SNLI.splits(self.text, self.labels)
        self.text.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name="840B", dim=300))
        self.labels.build_vocab(self.train)
        if not cfg.exp.cross_test:
            self.tr_iter, self.dev_iter, self.tst_iter = \
                data.BucketIterator.splits((self.train, self.dev, self.test),
                                           batch_sizes=[cfg.train.batch_size] * 3,
                                           device=device)

    def get_vocab(self):
        return self.text.vocab

class EvalDataPrep():
    '''
    Preparing a dataset for evaluating: loads dataset (MNLI, SNLI or SICK),
        splits it into sets. It's used for cross-dataset testing
        Sick dataset is found in https://github.com/felipessalvatore/NLI_datasets
    '''
    def __init__(self, cfg, text_field, label_field, device):
        if cfg.eval.dataset == 'mnli':
            self.train, self.dev, self.test = datasets.MultiNLI.splits(text_field, label_field)
            self.tr_iter, self.dev_iter, self.tst_iter = \
                data.BucketIterator.splits((self.train, self.dev, self.test),
                                           batch_sizes=[cfg.train.batch_size] * 3,
                                           device=device)

        if cfg.eval.dataset == 'snli':
            self.train, self.dev, self.test = datasets.SNLI.splits(text_field, label_field)
            self.tr_iter, self.dev_iter, self.tst_iter = \
                data.BucketIterator.splits((self.train, self.dev, self.test),
                                           batch_sizes=[cfg.train.batch_size] * 3,
                                           device=device)

        if cfg.eval.dataset == 'sick':
            self.train, self.dev, self.test = data.TabularDataset.splits(
                path=cfg.eval.sick_path,
                train='train_complete.csv',
                validation='train_complete.csv',
                test='train_complete.csv',
                format='csv',
                fields=[('premise', text_field),
                        ('hypothesis', text_field),
                        ('label', label_field)
                        ],
                skip_header=True)
            self.tr_iter, self.dev_iter, self.tst_iter = \
                data.BucketIterator.splits((self.train, self.dev, self.test),
                                           batch_sizes=[cfg.train.batch_size] * 3,
                                           device=device, sort_key=lambda ex: data.interleave_keys(
                        len(ex.premise), len(ex.hypothesis)))



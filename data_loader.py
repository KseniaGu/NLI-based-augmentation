from torchtext import data, datasets
from torchtext.vocab import GloVe

class DataPrep():
    def __init__(self, cfg): # cfg.train
        self.text = data.Field(lower=True, tokenize='spacy', batch_first=True)
        self.labels = data.Field(sequential=False, is_target=True, unk_token=None)
        if cfg.dataset == 'mnli':
            self.train, self.dev, self.test = datasets.MultiNLI.splits(self.text, self.labels)
        if cfg.dataset == 'snli':
            self.train, self.dev, self.test = datasets.SNLI.splits(self.text, self.labels)
        self.text.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name="840B", dim=300))
        self.labels.build_vocab(self.train)
        self.tr_iter, self.dev_iter, self.tst_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[cfg.batch_size] * 3,
                                       device=cfg.device)

    def get_vocab(self):
        return self.text.vocab

class EvalDataPrep():
    def __init__(self, cfg): #cfg.eval
        if cfg.dataset == 'snli':
            self.text = data.Field(lower=True, tokenize='spacy', batch_first=True)
            self.labels = data.Field(sequential=False, is_target=True, unk_token=None)
            self.train, self.dev, self.test = datasets.SNLI.splits(self.text, self.labels)

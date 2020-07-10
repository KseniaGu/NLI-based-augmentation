from torchtext import data, datasets
import os


# parameters:
DATASET = 'sick' # one of evals

# paths:
DATASET_PATH_DIR = os.path.join(os.path.split(os.getcwd())[0],'.data')


class DatasetLoader():
    def __init__(self):
        self.text_field = data.Field(tokenize='spacy', batch_first=True)  # lower=True
        self.label_field = data.Field(sequential=False, is_target=True, unk_token=None, batch_first=True, lower=True)

    def RteSplits(self):
        train, dev, test = data.TabularDataset.splits(
            path=os.path.join(DATASET_PATH_DIR, 'rte'),
            train='train_dev.csv',
            validation='train_dev.csv',
            test='train_dev.csv',
            format='csv',
            fields=[('premise', self.text_field),
                    ('hypothesis', self.text_field),
                    ('label', self.label_field)],
            skip_header=True)
        return train, dev, test

    def SickSplits(self, cor=False):
        if cor:
            fields = {'sentence_A': ('premise', self.text_field),
                      'sentence_B': ('hypothesis', self.text_field),
                      'entailment_AB': ('label', self.label_field)}
        else:
            fields = [('premise', self.text_field),
                      ('hypothesis', self.text_field),
                      ('label', self.label_field)]

        train, dev, test = data.TabularDataset.splits(
            path=os.path.join(DATASET_PATH_DIR, 'sick'),
            train='SICK_corrected.tsv' if cor else 'train_dev.csv',
            validation='SICK_corrected.tsv' if cor else 'train_dev.csv',
            test='SICK_corrected.tsv' if cor else 'train_dev.csv',
            format='tsv' if cor else 'csv',
            fields=fields,
            skip_header=False if cor else True)

        return train, dev, test

    def NliSplits(self, dataset):
        if dataset == 'mnli':
            return datasets.MultiNLI.splits(self.text_field, self.label_field)
        else:
            return datasets.SNLI.splits(self.text_field, self.label_field)

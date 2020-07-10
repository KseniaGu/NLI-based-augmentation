import torch
import re
import time
import gc
import os
from triggers_and_gender_bias.data_loader import DatasetLoader


# parameters:
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EVAL_DATASETS = ('snli','mnli','sick','sick_cor','rte')

# paths:
GEND_SYN_PATH = os.path.join(os.path.split(os.getcwd())[0], 'triggers_and_gender_bias/gender_synonyms')


def get_synonyms():
    f_syn, m_syn = [], []

    with open(os.path.join(GEND_SYN_PATH, 'female_list.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            f_syn.append(line[:-1])

    with open(os.path.join(GEND_SYN_PATH, 'male_list.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            m_syn.append(line[:-1])

    return f_syn, m_syn



class Roberta():
    def __init__(self):
        self._roberta_build()
        self.results = dict(zip(EVAL_DATASETS, (0,) * len(EVAL_DATASETS)))
        self.data_loader = DatasetLoader()
        self.l_maps = {'rte': {"entailment": 2, 'not_entailment': 1},
                       'sick': {"contradiction": 0, "neutral": 1, "entailment": 2},
                       'sick_cor': {"a_contradicts_b": 0, "a_neutral_b": 1, "a_entails_b": 2},
                       'snli': {"contradiction": 0, "neutral": 1, "entailment": 2},
                       'mnli': {"contradiction": 0, "neutral": 1, "entailment": 2}}

    def _roberta_build(self):
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
        if torch.cuda.is_available():
            self.roberta.cuda()
        self.roberta.eval()

    def _gender_change(self, sentence, i):
        regex = r'(\b[Mm][ae]n\b|\b[Bb]oys?\b)'
        old_sent = sentence
        finds = re.findall(regex, sentence)
        map_gen = {"Man": "Woman", "Men": "Woman",
                   'man': "woman", 'men': 'women',
                   'Boy': 'Girl', 'boy': 'girl'}

        for f in finds:
            f_cr = re.findall(r'[^s]*', f)[0]  # drops 's' if boys is found
            sentence = re.sub(f_cr, map_gen[f_cr], sentence)

        if old_sent != sentence:
            self.g_changes.append((i, old_sent, sentence))
        return sentence

    def _data_load(self, cfg_dataset):
        if cfg_dataset.startswith('sick'):
            return self.data_loader.SickSplits(cfg_dataset == 'sick_cor')
        elif cfg_dataset == 'rte':
            return self.data_loader.RteSplits()
        elif cfg_dataset.endswith('nli'):
            return self.data_loader.NliSplits(cfg_dataset)

    def _data_prep(self, cfg_dataset, g_change, trigger):
        self.g_changes = []
        self.dataset = []
        self.l_map = self.l_maps[cfg_dataset]
        train, dev, test = self._data_load(cfg_dataset)

        for i, exmp in enumerate(test):
            prem = ' '.join(exmp.premise) if exmp.premise[-1] == '.' else ' '.join(exmp.premise) + '.'
            hyp = ' '.join(exmp.hypothesis) if exmp.hypothesis[-1] == '.' else ' '.join(exmp.hypothesis) + '.'
            if trigger:
                hyp = 'Nobody ' + hyp[0].lower() + hyp[1:]

            label = exmp.label
            if g_change:
                self.dataset.append(
                    [self.roberta.encode(self._gender_change(prem, i), self._gender_change(hyp, i)), self.l_map[label]])
            else:
                self.dataset.append([self.roberta.encode(prem, hyp), self.l_map[label]])
            if i % 100 == 0:
                print(i)

    def make_eval(self, cfg_dataset, g_change=False, trigger=False):
        self._data_prep(cfg_dataset, g_change, trigger)
        self.rev_l_map = {y: x for x, y in self.l_map.items()}

        print("Eval started")
        t = time.time()
        wrong_predicts = []
        correct_predicts, nrof_samples = 0, 0

        for i, data in enumerate(self.dataset):
            prem_hyp, label = data  # ?
            prediction = self.roberta.predict('mnli', prem_hyp).argmax().item()
            if cfg_dataset == 'rte':
                prediction = torch.where(torch.tensor(prediction, dtype=torch.int64) > 0,
                                         torch.tensor(prediction, dtype=torch.int64), torch.tensor(1,
                                                                                                   dtype=torch.int64))  # torch.ones(prediction.shape,dtype=torch.int64))
                comp = int(int(prediction) == label)
                if comp == 0:
                    wrong_predicts.append(
                        (i, self.roberta.decode(prem_hyp), self.rev_l_map[label], self.rev_l_map[int(prediction)]))
                correct_predicts += comp
            else:
                comp = int(prediction == label)
                if comp == 0:
                    wrong_predicts.append(
                        (i, self.roberta.decode(prem_hyp), self.rev_l_map[label], self.rev_l_map[prediction]))
                correct_predicts += comp

            nrof_samples += 1
            if i % 100 == 0:
                print(i)

        t = time.time() - t
        acc = float(correct_predicts) / nrof_samples
        self.results[cfg_dataset] = (acc, " Time taken: {:.3f}".format(t))
        print("acc: {}, time taken: {:.3f}".format(acc, t))
        gc.collect()
        return wrong_predicts

    def get_features(self, sent):
        doc = self.roberta.extract_features_aligned_to_words(sent)
        return doc

    def evaluate_raw_seq(self, prem, hyp, label):
        rev_l_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
        encoded_sents = self.roberta.encode(prem, hyp)
        prediction = self.roberta.predict('mnli', encoded_sents).argmax()

        print("Premise: {}\nHypothesis: {}\nLabel: {}\nPredicted: {}\n\n".format(
            prem, hyp, label, rev_l_map[prediction.cpu().numpy().tolist()]))

        return prediction

    def count_gender(self, cfg_dataset, tr=False):
        train, dev, test = self._data_load(cfg_dataset)
        fm_counts = [0, 0]
        f_syn, m_syn = get_synonyms()
        dataset = train if tr else test

        for i, data in enumerate(dataset):
            prem = ' '.join(data.premise)
            hyp = ' '.join(data.hypothesis)
            sent = prem + ' ' + hyp

            for i, syn in enumerate((f_syn, m_syn)):
                for s in syn:
                    regex = r'\b[' + s[0].upper() + s[0].lower() + ']' + s[1:] + r's?\b'
                    finds = re.findall(regex, sent)
                    if finds:
                        print("f {}\nSent: {}\n\n".format(finds, sent))
                    fm_counts[i] += len(finds)

        return fm_counts
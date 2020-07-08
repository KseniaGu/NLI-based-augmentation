import torch
import torchtext
from nltk.corpus import stopwords
import random


#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# manually as certain simbols are not presented in lib's tool
puncts = ('.', ',','..','...','!','?',' ','-',':','—','(',')',"'")



# parameters
DATASET_NAME = 'rte'
DATASET_PATH = '.data/rte'


class DataPrep():
    def __init__(self):
        self.text = torchtext.data.Field(lower=False, tokenize='spacy', batch_first=True)
        self.labels = torchtext.data.Field(sequential=False, is_target=True, unk_token=None)

        if DATASET_NAME  == 'rte':
            self.train, self.dev, self.test = torchtext.data.TabularDataset.splits(
                path=DATASET_PATH,
                train='train.tsv',
                validation='dev.tsv',
                test='dev.tsv',
                format='tsv',
                fields=[('index', None),
                        ('sentence1', self.text),
                        ('sentence2', self.text),
                        ('label', self.labels)
                        ],
                skip_header=True)


class Augment():
    def __init__(self):
        self.new_overlap_changes = []
        self.start = len(self.new_overlap_changes)
        self._build_roberta()
        self.counter = 0

    def _build_roberta(self):
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.roberta.cuda()
        self.roberta.eval()

    def mask_sentence(self, sentence, word_ind):
        new_sent = sentence[:]
        new_sent[word_ind] = '<mask>'
        new_sent = ' '.join(new_sent)
        new_sent = new_sent[0].upper() + new_sent[1:]
        new_sent = new_sent + '.' if new_sent[-1] not in puncts else new_sent
        return new_sent

    def augment(self, train_set):
        drop_space = lambda word: word[1:] if word.startswith(' ') else word

        for i, data in enumerate(train_set[self.start:]):
            self.new_overlap_changes.append([])
            p_sent = data.sentence1  # uncased
            h_sent = data.sentence2  # uncased
            p_sent_lower = [x.lower() for x in p_sent]
            h_sent_lower = [x.lower() for x in h_sent]
            overlap = list(set(p_sent_lower) & set(h_sent_lower))
            words_to_change = [x for x in overlap if x not in puncts and x not in stop_words and p_sent_lower.count(
                x) == 1 and h_sent_lower.count(x) == 1]

            if words_to_change:
                while len(words_to_change) != 0:  # and len(self.new_overlap_changes[-1])<2:
                    word_num = random.randint(0, len(words_to_change) - 1)
                    old_word = words_to_change[word_num]  # cased
                    p_old_w_ind = p_sent_lower.index(old_word)
                    h_old_w_ind = h_sent_lower.index(old_word)
                    masked_p_sent = self.mask_sentence(p_sent, p_old_w_ind)
                    masked_h_sent = self.mask_sentence(h_sent, h_old_w_ind)
                    p_result = self.roberta.fill_mask(masked_p_sent, topk=20)
                    h_result = self.roberta.fill_mask(masked_h_sent, topk=20)
                    p_words = dict([(drop_space(x[2]).lower(), (x[1], k)) for k, x in enumerate(p_result) if
                                    drop_space(x[2]).lower() != old_word and drop_space(x[2]) not in puncts])
                    h_words = dict([(drop_space(x[2]).lower(), (x[1], k)) for k, x in enumerate(h_result) if
                                    drop_space(x[2]).lower() != old_word and drop_space(x[2]) not in puncts])
                    res_overlap = list(set(p_words.keys()) & set(h_words.keys()))

                    sort_fn = lambda x: p_words[x][0] + h_words[x][0]
                    sorted_res_overlap = sorted(res_overlap, key=sort_fn, reverse=True)

                    if res_overlap:
                        new_words = []
                        for res in sorted_res_overlap[:4]:
                            p_result_ind = p_words[res][1]
                            new_words.append(p_result[p_result_ind][2])
                        self.new_overlap_changes[-1].append((p_old_w_ind, h_old_w_ind, new_words))
                        self.counter += 1
                    words_to_change.remove(old_word)

            if len(self.new_overlap_changes[-1]) == 0:
                self.new_overlap_changes[-1].append((-1, -1, ['_']))


            if ((i + self.start) % 1000) == 0:
                print('Premise: {}\nHypothesis: {}\nIndexes: {}'.format(data.sentence1, data.sentence2,self.new_overlap_changes[-1]))
                # changes_copy = self.new_overlap_changes[:] # for saving

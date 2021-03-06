import os
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import Dataset
from finlp.data.model import Simple
class CoNLLDataset(Dataset):

    def __init__(self, data_file, build_vocab=False):
        self.data_file = data_file
        self.sentences = []
        self.tags = []
        self.samples = []
        self.build_dataset()
        if build_vocab:
            self.build_vocab_from_sample()

    def build_dataset(self):
        assert os.path.exists(self.data_file)
        self.parse_data_from_file()
        self.convert_to_samples()

    def parse_data_from_file(self):
        sentences = []
        sen_tags = []
        words, labels = [], []
        with open(self.data_file,'r',encoding='utf-8') as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if len(words) == 0:
                        continue
                    
                    sentences.append(words)
                    sen_tags.append(labels)
                    words = []
                    labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")

            if words:
                sentences.append(words)
                sen_tags.append(labels)
        self.sentences = sentences
        self.tags = sen_tags

    def convert_to_samples(self):
        for idx in range(len(self.sentences)):
            self.samples.append(Simple(
                words=self.sentences[idx],
                tags=self.tags[idx]
            ))
        
    def build_vocab_from_sample(self):
        tokens = []
        for sample in self.samples:
            tokens += sample.words
        counter = Counter(tokens)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.vocab = vocab(ordered_dict, specials=['<unk>','<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def load_tags(labels_file):
    tags = []
    with open(labels_file,'r',encoding='utf-8') as f:
        for line in f:
            tags.append(line.replace("\n", ""))
    tags += ['<pad>']
    tag2id = {tag: idx for idx,tag in enumerate(tags)}
    id2tag = {idx: tag for idx,tag in enumerate(tags)}
    return tag2id, id2tag
    
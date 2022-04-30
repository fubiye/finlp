import os
from torch.utils.data import Dataset
from finlp.data.model import Simple
class CoNLLDataset(Dataset):

    def __init__(self, data_file):
        self.data_file = data_file
        self.sentences = []
        self.tags = []
        self.samples = []
        self.build_dataset()
        

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
    
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from finlp.data.tokenizer import Tokenizer

class NerDataLoader():

    def __init__(self, dataset, tokenizer: Tokenizer, tag2id, batch_size=16):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, samples):
        
        words = [self.tokenizer.tokenize(sample.words) for sample in samples]
        tags = [[self.tag2id[tag] for tag in sample.tags] for sample in samples]
        
        pad_token_idx = self.tokenizer.vocab['<pad>']
        padded_words, seq_lengths = pad_sequence(words, padding_value=pad_token_idx)  # T * B * n
        pad_tag_idx = tag2id['<pad>']
        padded_tags, seq_lengths = pad_sequence(tags, padding_value=pad_tag_idx)

        return {
            'padded_words': padded_words,
            'padded_tags': padded_tags,
            'seq_lengths': seq_lengths,
        }
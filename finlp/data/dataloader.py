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
        seq_lengths = []
        batch_tokens = []
        batch_word_ids = []
        batch_tags = []
        for sample in samples:
            tokens, word_ids = self.tokenizer.tokenize(sample.words)
            seq_lengths.append(len(sample.words))
            batch_tokens.append(torch.LongTensor(tokens))
            batch_word_ids.append(word_ids)
            
            tag_ids = torch.LongTensor([self.tag2id[tag] for tag in sample.tags])
            batch_tags.append(tag_ids)

        
        pad_token_idx = self.tokenizer.vocab['<pad>']
        padded_words  = pad_sequence(batch_tokens, padding_value=pad_token_idx)  # T * B * n
        pad_tag_idx = self.tag2id['<pad>']
        padded_tags  = pad_sequence(batch_tags, padding_value=pad_tag_idx)

        return {
            'padded_tokens': padded_words,
            'batch_word_ids': batch_word_ids,
            'padded_tags': padded_tags,
            'seq_lengths': seq_lengths,
        }
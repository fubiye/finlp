import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def get_data_loader(dataset, batch_size=16, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

def rnn_collate_fn(samples):
    words = [seq_tokenizer.tokenize(sample.words) for sample in samples]
    tags = [sample.tags for sample in samples]
    
    padded_words = pad_sequence(words)  # T * B * n
    return {
        
    }

class NerDataLoader():

    def __init__(self, dataset, seq_tokenizer, tag_tokenizer, batch_size=16):
        self.dataset = dataset
        self.seq_tokenizer = seq_tokenizer
        self.tag_tokenizer = tag_tokenizer
        self.loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, samples):
        
        words = [self.seq_tokenizer.tokenize(sample.words) for sample in samples]
        tags = [seq_tokenizer.sample.tags for sample in samples]
        
        padded_words = pad_sequence(words)  # T * B * n
        return {
            
        }
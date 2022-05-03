import torch
from torch.utils.data import DataLoader

from finlp.data.model import Tokenizer


class BertDataLoader:

    def __init__(self, dataset, tokenizer: Tokenizer, tag2id,shuffle=False,batch_size=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.tag_pad_id = tag2id.get('<pad>')
        self.loader = DataLoader(dataset,shuffle=shuffle,batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, samples):
        batch_words = []
        batch_tags = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        word_ids = []
        token_tag_ids = []
        for sample in samples:
            batch_words.append(sample.words)
            tag_ids = torch.LongTensor([self.tag2id[tag] for tag in sample.tags])
            batch_tags.append(tag_ids)

        tokens = self.tokenizer.tokenize(batch_words)
        for idx, token in enumerate(tokens):
            input_ids.append(token['input_ids'])
            token_type_ids.append(token['token_type_ids'])
            attention_mask.append(token['attention_mask'])
            word_ids.append(token['word_ids'])
            sample_token_tag_ids = assign_token_tag_ids(token['word_ids'], batch_tags[idx], self.tag_pad_id)
            token_tag_ids.append(sample_token_tag_ids)
        return {
            'batch_tags': batch_tags,
            'input_ids': torch.stack(input_ids),
            'token_type_ids': torch.stack(token_type_ids),
            'attention_mask': torch.stack(attention_mask),
            'word_ids': word_ids,
            'token_tag_ids': torch.LongTensor(token_tag_ids)
        }


def assign_token_tag_ids(word_ids, tag_ids, pad_idx=-100):
    token_tag_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            token_tag_ids.append(pad_idx)
        elif word_id == previous_word_id:
            token_tag_ids.append(pad_idx)
        else:
            token_tag_ids.append(tag_ids[word_id])
        previous_word_id = word_id
    return token_tag_ids


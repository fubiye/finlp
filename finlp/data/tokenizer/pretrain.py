from transformers import AutoTokenizer

from finlp.data.model import Tokenizer

class PreTrainTokenizer(Tokenizer):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, batch_sentences):
        samples = []
        tokenized = self.tokenizer(batch_sentences,padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
        for idx, (input_ids, token_type_ids, attention_mask) in enumerate(
                zip(tokenized['input_ids'], tokenized['token_type_ids'],
                    tokenized['attention_mask'])):
            word_ids = tokenized.word_ids(batch_index=idx)
            samples.append({
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'word_ids': word_ids
            })
        return samples

from finlp.data.model import Tokenizer

class VocabTokenizer(Tokenizer):
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def tokenize(self, words):
        tokens = []
        word_ids = []
        for idx, word in enumerate(words):
            tokens.append(self.vocab[word])
            word_ids.append(idx)

        return tokens, word_ids


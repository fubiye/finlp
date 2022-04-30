class Tokenizer(object):

    def tokenize(self, words):
        tokens = []
        word_ids = []
        for idx, word in enumerate(words):
            tokens.append(word)
            word_ids.append(idx)

        return tokens, word_ids

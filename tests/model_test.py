import unittest
from finlp.data.model import Tokenizer

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenzier = Tokenizer()
       
    def test_tokenize(self):
        words = ['hello','world']
        tokens, word_ids = self.tokenzier.tokenize(words)
        self.assertEquals(len(tokens), 2)
        for idx, word_id in enumerate(word_ids):
            self.assertEquals(idx, word_id)

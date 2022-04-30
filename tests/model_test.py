import unittest
from finlp.data.model import Tokenizer, Simple

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenzier = Tokenizer()
       
    def test_tokenize(self):
        words = ['hello','world']
        tokens, word_ids = self.tokenzier.tokenize(words)
        self.assertEquals(len(tokens), 2)
        for idx, word_id in enumerate(word_ids):
            self.assertEquals(idx, word_id)

class SampleTest(unittest.TestCase):

    def test_sample(self):
        words = ["OMERS","is","one","of","Canada","'","s","largest","pension","funds","with","over","$","85",".","2","billion","of","net","assets","as","of","year","-","end","2016","."]
        tags = ['B-ORG','O','O','O','B-LOC','O','O','O','O','O','O','O','B-MONEY','I-MONEY','I-MONEY','I-MONEY','I-MONEY','O','O','O','O','O','B-DATE','I-DATE','I-DATE','I-DATE','O']

        sample = Simple(words, tags)
        self.assertEquals(27, len(words))
        self.assertEquals(27, len(tags))
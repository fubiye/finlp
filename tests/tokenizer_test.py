import unittest
from transformers import AutoTokenizer
from finlp.data.tokenizer.pretrain import PreTrainTokenizer

class PretrainTokenizerTest(unittest.TestCase):

    def test_tokenize(self):
        _tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = PreTrainTokenizer(_tokenizer)
        words = [["OMERS", "is", "one", "of", "Canada", "'", "s", "largest", "pension", "funds", "with", "over", "$",
                 "85", ".", "2", "billion", "of", "net", "assets", "as", "of", "year", "-", "end", "2016", "."],
                 ["Oxford", "Properties", "Group", "is", "the", "global", "real", "estate", "investment", "arm", "of",
                  "OMERS", "."]
                 ]
        tokenized = tokenizer.tokenize(words)
        print(tokenized)


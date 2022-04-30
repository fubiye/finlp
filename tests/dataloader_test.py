import os 
import unittest
from finlp.data.dataset import CoNLLDataset
from finlp.data.dataloader import get_data_loader, rnn_collate_fn
from finlp.data.tokenizer import VocabTokenizer

dir_path = os.path.dirname(os.path.abspath(__file__))

class DataLoaderTest(unittest.TestCase):

    def test_dataloader(self):
        data_file = os.path.join(dir_path, 'resources/train.txt')
        dataset = CoNLLDataset(data_file, build_vocab=True)
        tokenizer = VocabTokenizer(dataset.vocab)
        loader = get_data_loader(dataset, batch_size=2, collate_fn=rnn_collate_fn)
        self.assertEquals(4, len(loader))
        first_batch = next(iter(loader))

import os
import unittest
from finlp.data.dataset import CoNLLDataset

dir_path = os.path.dirname(os.path.abspath(__file__))

class DatasetTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_when_file_not_exists_then_failed(self):
        data_file = os.path.join(dir_path, 'resources/not_exists.txt')
        dataset = CoNLLDataset(data_file)

    def test_load_files(self):
        data_file = os.path.join(dir_path, 'resources/train.txt')
        dataset = CoNLLDataset(data_file)
        self.assertEquals(8, len(dataset.sentences))
        self.assertEquals(8, len(dataset.tags))

        self.assertEquals(9, len(dataset.sentences[0]))
        self.assertEquals(9, len(dataset.tags[0]))

        self.assertEquals(8, len(dataset))
        self.assertEquals(9, len(dataset[0].words))
        
    def test_build_vocab(self):
        data_file = os.path.join(dir_path, 'resources/train.txt')
        dataset = CoNLLDataset(data_file, build_vocab=True)
        vocab = dataset.vocab
        self.assertEquals(93, len(vocab))
        self.assertEquals(0, vocab['<unk>'])
        self.assertEquals(1, vocab['<pad>'])
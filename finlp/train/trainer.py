from finlp.model.bilstm_crf import BiLstmModel

class BiLstmTrainer():

    def __init__(self, train_data, dev_data, test_data,
                    word2id, tag2id):
        self.train_sents, self.train_tags = train_data
        self.dev_sents, self.dev_tags = dev_data
        self.test_sents, self.test_tags = test_data

        self.word2id = word2id
        self.tag2id = tag2id

        vocab_size = len(word2id)
        output_size = len(tag2id)
        self.model = BiLstmModel(vocab_size, output_size)

        
    def train(self):
        self.model.train(
            self.train_sents, self.train_tags, 
            self.dev_sents, self.dev_tags, 
            self.word2id, self.tag2id
        )


from finlp.dataset.conll2003 import CoNLL2003
from finlp.vocab.util import add_special_token, add_bert_special_token
from finlp.train.trainer import BiLstmTrainer
from finlp.train.bert_trainer import BertTrainer

def main():
    print("loading data...")
    train_sentences, train_tags, word2id, tag2id = CoNLL2003(split='train')
    dev_sentences, dev_tags = CoNLL2003(split='dev', make_vocab=False)
    test_sentences, test_tags = CoNLL2003(split='test', make_vocab=False)
    print(f'loaded train sentences:{len(train_sentences)}')

    print("start train and eval BiLSTM model...")
    bilstm_word2id, bilstm_tag2id = add_special_token(word2id, tag2id, crf=False)
    bilstm_trainer = BiLstmTrainer(
        (train_sentences, train_tags),
        (dev_sentences, dev_tags), 
        (test_sentences, test_tags), 
        word2id, tag2id
    )
    bilstm_trainer.train()

    # print("start train and eval Bert Model...")
    # bert_tag2id = add_bert_special_token(tag2id)
    # bert_trainer = BertTrainer(
    #     (train_sentences, train_tags),
    #     (dev_sentences, dev_tags), 
    #     (test_sentences, test_tags), 
    #     word2id, bert_tag2id)
    # bert_trainer.train()

if __name__ == '__main__':
    main()
    
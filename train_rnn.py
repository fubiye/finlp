import os

from finlp.data.dataset import CoNLLDataset, load_tags
from finlp.data.tokenizer.vocab import VocabTokenizer
from finlp.data.loader.vocab import NerDataLoader
from finlp.train.rnn_trainer import RnnTrainer

def main():
    print("bootstraping application...")
    train_dataset = CoNLLDataset(train_file, build_vocab=True)
    dev_dataset = CoNLLDataset(dev_file)
    test_dataset = CoNLLDataset(test_file)

    tag2id, id2tag = load_tags(labels_file)
    tokenizer = VocabTokenizer(train_dataset.vocab)
    train_loader = NerDataLoader(train_dataset, tokenizer, tag2id)
    dev_loader = NerDataLoader(dev_dataset, tokenizer, tag2id)
    test_loader = NerDataLoader(test_dataset, tokenizer, tag2id)
    trainer = RnnTrainer(train_loader.loader, dev_loader.loader, test_loader.loader, tokenizer, tag2id, id2tag)
    trainer.train()
    trainer.test()

if __name__ == "__main__":
    dataset = 'conll2003'
    train_file = os.path.expanduser(f"~/.cache/{dataset}/train.txt")
    dev_file = os.path.expanduser(f"~/.cache/{dataset}/dev.txt")
    test_file = os.path.expanduser(f"~/.cache/{dataset}/test.txt")
    labels_file = os.path.expanduser(f"~/.cache/{dataset}/labels.txt")
    main()
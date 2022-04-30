import os
from finlp.data.dataset import CoNLLDataset, load_tags
from finlp.data.tokenizer import VocabTokenizer
from finlp.data.dataloader import NerDataLoader
from finlp.train.rnn_trainer import RnnTrainer

def main():
    print("bootstraping application...")
    train_dataset = CoNLLDataset(train_file, build_vocab=True)
    dev_dataset = CoNLLDataset(dev_file)
    tag2id, id2tag = load_tags(labels_file)
    tokenizer = VocabTokenizer(train_dataset.vocab)
    train_loader = NerDataLoader(train_dataset, tokenizer, tag2id)
    dev_loader = NerDataLoader(dev_dataset, tokenizer, tag2id)
    trainer = RnnTrainer(train_loader.loader, dev_loader.loader, tokenizer, tag2id)
    trainer.train()

if __name__ == "__main__":
    train_file = os.path.expanduser("~/.cache/conll2003/train.txt")
    dev_file = os.path.expanduser("~/.cache/conll2003/dev.txt")
    labels_file = os.path.expanduser("~/.cache/conll2003/labels.txt")
    main()
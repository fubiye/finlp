import os

from finlp.data.dataset import CoNLLDataset, load_tags
from finlp.data.tokenizer.pretrain import PreTrainTokenizer
from finlp.data.loader.pretrain import BertDataLoader
from finlp.train.transformer import TransformersTrainer

def main():
    print("bootstraping application...")
    train_dataset = CoNLLDataset(train_file)
    dev_dataset = CoNLLDataset(dev_file)
    test_dataset = CoNLLDataset(test_file)
    tag2id, id2tag = load_tags(labels_file)

    tokenizer = PreTrainTokenizer(model_name)
    train_loader = BertDataLoader(train_dataset, tokenizer, tag2id)
    dev_loader = BertDataLoader(dev_dataset, tokenizer, tag2id)
    test_loader = BertDataLoader(test_dataset, tokenizer, tag2id)
    trainer = TransformersTrainer(model_name, train_loader.loader, dev_loader.loader, test_loader.loader, tokenizer, tag2id, id2tag)
    trainer.train()
    trainer.test()

if __name__ == "__main__":
    model_name = 'bert-base-uncased'
    dataset = 'conll2003'
    train_file = os.path.expanduser(f"~/.cache/{dataset}/train.txt")
    dev_file = os.path.expanduser(f"~/.cache/{dataset}/dev.txt")
    test_file = os.path.expanduser(f"~/.cache/{dataset}/test.txt")
    labels_file = os.path.expanduser(f"~/.cache/{dataset}/labels.txt")
    main()

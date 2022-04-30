import os
from finlp.data.dataset import CoNLLDataset, load_tags
from finlp.data.tokenizer import VocabTokenizer
from finlp.data.dataloader import NerDataLoader

def main():
    print("bootstraping application...")
    train_dataset = CoNLLDataset(train_file, build_vocab=True)
    tag2id, id2tag = load_tags(labels_file)
    tokenizer = VocabTokenizer(train_dataset.vocab)
    train_loader = NerDataLoader(train_dataset, tokenizer, tag2id)

    for idx, batch in enumerate(train_loader.loader):
        print(f'batch: {idx}')

if __name__ == "__main__":
    train_file = os.path.expanduser("~/.cache/conlldev/train.txt")
    labels_file = os.path.expanduser("~/.cache/conlldev/labels.txt")
    main()
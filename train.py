import os
from finlp.data.dataset import CoNLLDataset
from finlp.data.tokenizer import VocabTokenizer

def main():
    print("bootstraping application...")
    train_dataset = CoNLLDataset(train_file, build_vocab=True)
    tokenizer = VocabTokenizer(train_dataset.vocab)

if __name__ == "__main__":
    train_file = os.path.expanduser("~/.cache/conlldev/train.txt")
    main()
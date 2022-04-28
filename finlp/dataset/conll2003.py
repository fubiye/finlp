import os
data_dir = os.path.expanduser('~/.cache/conlldev')

def CoNLL2003(split:str, make_vocab=True, data_dir=data_dir):
    assert split in ['train', 'dev','test']

    sentences = []
    sen_tags = []
    words, labels = [], []
    with open(os.path.join(data_dir, split+'.txt'),'r',encoding='utf-8') as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if len(words) == 0:
                    continue
                
                sentences.append(words)
                sen_tags.append(labels)
                words = []
                labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")

        if words:
            sentences.append(words)
            sen_tags.append(labels)
        

    if make_vocab:
        word2id = build_map(sentences)
        tag2id = build_map(sen_tags)
        return sentences, sen_tags, word2id, tag2id
    return sentences, sen_tags

def build_map(lists):
    vocab = {}
    for seq in lists:
        for ele in seq:
            if ele not in vocab:
                vocab[ele] = len(vocab)

    return vocab

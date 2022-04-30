
# LSTM requires PAD and UNK
# LSTM-CRF requires <start> and <end> when decoding
def add_special_token(word2id, tag2id, crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)

    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)

    if crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'](word2id)

        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)
    return word2id, tag2id

def add_bert_special_token(tag2id):
    tag2id['<pad>']  = -100
    return tag2id

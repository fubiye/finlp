import torch 
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BiLstm(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):

        super(BiLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, sents, lengths):
        embdded = self.embedding(sents) # [batch_size, padded_seq_len, embedding_dim]
        packed = pack_padded_sequence(embdded, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        # lstm_out: [batch_size, padded_seq_len, hidden_size * 2]
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.linear(lstm_out) # [batch_size, padded_seq_len, output_size]
        return logits
    
    def test(self, sents, lengths, _):
        logits = self.forward(sents, lengths)
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids

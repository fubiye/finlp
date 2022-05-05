import torch 
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF

class BiLstm(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):

        super(BiLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, sents, lengths):
        embdded = self.embedding(sents) # [batch_size, padded_seq_len, embedding_dim]
        packed = pack_padded_sequence(embdded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        # lstm_out: [batch_size, padded_seq_len, hidden_size * 2]
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.linear(lstm_out) # [batch_size, padded_seq_len, output_size]
        return logits

class BiLstmCrf(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, pad_tag_id=9):
        super(BiLstmCrf, self).__init__()
        self.pad_tag_id = pad_tag_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, output_size)
        self.crf = CRF(num_tags=output_size, batch_first=True)


    def forward(self, sents, lengths, labels):
        embdded = self.embedding(sents)  # [batch_size, padded_seq_len, embedding_dim]
        packed = pack_padded_sequence(embdded, lengths, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        # lstm_out: [batch_size, padded_seq_len, hidden_size * 2]
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.linear(lstm_out)  # [batch_size, padded_seq_len, output_size]
        mask = labels != self.pad_tag_id
        mask = mask.byte()
        _labels = torch.clamp_max(labels,self.pad_tag_id-1)
        loss = self.crf(logits, _labels, mask=mask, reduction='mean')
        results = self.crf.decode(logits)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result))
        foward_out = torch.stack(result_tensor)
        return -loss, foward_out.to(labels.device)
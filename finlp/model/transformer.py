from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BertNerModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertNerModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        mask = (attention_mask == 1)
        sequence_output = sequence_output[mask]
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        return logits

class BertLstmModel(BertPreTrainedModel):

    def __init__(self, config, hidden_size):
        super(BertLstmModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, config.num_labels)
        self.init_weights()

    def forward(self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
            ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        # mask = (attention_mask == 1)
        # sequence_output = sequence_output[mask]
        sequence_output = self.dropout(sequence_output)
        seq_lengths = [len(mask[mask == 1]) for mask in attention_mask]
        packed = pack_padded_sequence(sequence_output, seq_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        # lstm_out: [batch_size, padded_seq_len, hidden_size * 2]
        # lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True,padding_value=9)
        logits = self.linear(lstm_out.data)
        return logits
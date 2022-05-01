from transformers import BertModel, BertPreTrainedModel

import torch.nn as nn

class BertNerModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertNerModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
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
        logits = self.linear(sequence_output)
        return logits
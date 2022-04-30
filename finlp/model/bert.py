from transformers import BertModel, BertPreTrainedModel
import torch 
import torch.nn as nn

from finlp.util.bert_util import valid_sequence_output

class BertSoftmaxForNerModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNerModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
        self.epoches = 1
        self.batch_size = 32

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

        outputs = (logits,) + outputs[2:] 


        return outputs
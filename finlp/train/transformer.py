import torch
from transformers import AutoConfig, AdamW

from finlp.loss.util import cross_entropy
from finlp.model.transformer import BertNerModel
from finlp.metrics.entity import EntityMetrics

class TransformersTrainer:
    def __init__(self, model_name, train_loader, dev_loader, test_loader, tokenizer, tag2id, id2tag):
        self.model_name = model_name
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.tag2id = tag2id
        self.id2tag = id2tag
        self.init_params()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = self.output_size
        self.model = BertNerModel(self.config)
        self.model.to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_fn = cross_entropy

    def init_params(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_size = len(self.tag2id) - 1  # remove <pad>
        self.epoches = 1
        self.lr = 1e-3
        self.weight_decay = 0.
        self.print_step = 5

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = self.model.bert.named_parameters()
        classifier_parameters = self.model.linear.named_parameters()

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.lr
            },
            {
                "params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.lr
            },
            {
                "params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.lr
            },
            {
                "params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.lr
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer

    def train(self):
        self.model.train()
        for epoch in range(self.epoches):
            losses = 0.
            step = 0
            for idx, batch in enumerate(self.train_loader):
                step += 1
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=attention_mask,
                    token_type_ids=batch['token_type_ids'].to(self.device))
                token_tag_ids = batch['token_tag_ids'].to(self.device)
                mask = (attention_mask == 1)
                token_tag_ids = token_tag_ids[mask]

                self.optimizer.zero_grad()
                loss = self.loss_fn(logits, token_tag_ids, self.tag2id)
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                losses += loss.item()
                if step % self.print_step == 0:
                    print("training epoch {}, steps: {} loss: {:.4f}".format(
                        epoch, step, losses / self.print_step
                    ))
                    losses = 0
            val_loss = self.validate()
            print('epoch {}, val loss: {:.4f}'.format(epoch, val_loss))

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            losses = 0.
            step = 0

            for idx, batch in enumerate(self.dev_loader):
                step += 1
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=attention_mask,
                    token_type_ids=batch['token_type_ids'].to(self.device))
                token_tag_ids = batch['token_tag_ids'].to(self.device)
                mask = (attention_mask == 1)
                token_tag_ids = token_tag_ids[mask]
                loss = self.loss_fn(logits, token_tag_ids, self.tag2id)
                losses += loss.item()

            val_loss = losses / step
            return val_loss

    def test(self):
        self.model.eval()
        with torch.no_grad():
            losses = 0.
            step = 0

            targets = []
            predicts = []

            for idx, batch in enumerate(self.test_loader):
                step += 1
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=attention_mask,
                    token_type_ids=batch['token_type_ids'].to(self.device))
                token_tag_ids = batch['token_tag_ids'].to(self.device)
                mask = (attention_mask == 1)
                token_tag_ids = token_tag_ids[mask]
                loss = self.loss_fn(logits, token_tag_ids, self.tag2id)
                losses += loss.item()

                batch_predicts = torch.argmax(logits, dim=2)
                targets.append(token_tag_ids)
                predicts.append(batch_predicts)

            test_loss = losses / step
            print("test loss: {}".format(test_loss))
            metrics = EntityMetrics(self.tag2id, self.id2tag)
            result = metrics.report(targets, predicts)
            metrics.print_result(result)

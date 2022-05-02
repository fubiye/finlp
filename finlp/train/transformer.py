import torch
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup

from finlp.loss.util import cross_entropy
from finlp.model.transformer import BertNerModel
from finlp.metrics.entity import EntityMetrics
from finlp.metrics.tag import TagMetrics

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
        self.optimizer, self.scheduler = self.get_optimizer()
        self.loss_fn = cross_entropy

    def init_params(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.output_size = len(self.tag2id) - 1  # remove <pad>
        self.epoches = 3
        self.bert_lr = 3e-5
        self.lr = 5e-4

        self.weight_decay = 0.
        self.print_step = 5
        self.max_grad_norm = 1.
        self.warmup_steps = 0
        self.adam_epsilon = 1e-8

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = self.model.bert.named_parameters()
        classifier_parameters = self.model.linear.named_parameters()

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.bert_lr
            },
            {
                "params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.bert_lr
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.bert_lr,eps=self.adam_epsilon)
        num_train_steps = len(self.train_loader) * self.epoches
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=num_train_steps)
        return optimizer, scheduler

    def train(self):
        torch.manual_seed(42)
        self.model.train()
        for epoch in range(self.epoches):
            losses = 0.
            step = 0
            batch_num = len(self.train_loader.dataset) / self.train_loader.batch_size
            eval_interval = int(batch_num * 0.3)
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                losses += loss.item()
                if step % self.print_step == 0:
                    print("training epoch {}, steps: {} loss: {:.4f}".format(
                        epoch, step, losses / self.print_step
                    ))
                    losses = 0
                if step % eval_interval == 0:
                    val_loss, val_result = self.validate()
                    print('epoch {}, val loss: {:.4f}, val f1: {:.4f}'.format(epoch, val_loss, val_result['macro_avg']['f1']))

            val_loss, val_result = self.validate()
            print('epoch {}, val loss: {:.4f}, val f1: {:.4f}'.format(epoch, val_loss, val_result['macro_avg']['f1']))
    def validate(self):
        return self.eval_step(self.dev_loader, 'val')

    def test(self):
        return self.eval_step(self.test_loader, 'test')

    def eval_step(self, loader, mode):
        self.model.eval()
        with torch.no_grad():
            losses = 0.
            step = 0

            targets = []
            predicts = []

            for idx, batch in enumerate(loader):
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
                dim = len(logits.shape) - 1
                batch_predicts = torch.argmax(logits, dim=dim)
                targets.append(token_tag_ids)
                predicts.append(batch_predicts)

            avg_loss = losses / step
            print("{} loss: {}".format(mode, avg_loss))
            # metrics = EntityMetrics(self.tag2id, self.id2tag)
            metrics = TagMetrics(self.tag2id, self.id2tag)
            result = metrics.report(targets, predicts)
            metrics.print_result(result)
            
            return avg_loss, result

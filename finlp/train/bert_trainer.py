import os
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig , AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from finlp.model.bert import BertSoftmaxForNerModel
from finlp.loss.util import cross_entropy

class BertTrainer():

    def __init__(self,train_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("start init bert model")
        cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')
        model_name = 'bert-base-uncased'

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = output_size
        self.model = BertSoftmaxForNerModel(self.config)
        self.model.to(self.device)

        self.epoches = 1
        self.batch_size = 8
        self.print_step = 5
        self.lr = 1e-3
        self.weight_decay = 0.
        
        self.loss_fn = cross_entropy
        self.optimizer = self.get_optimizer()

        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

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
        dataset = BertNerDataset(self.tokenizer, self.train_sents, self.train_tags, self.tag2id)
        dataloder = DataLoader(dataset,batch_size = self.batch_size,shuffle=True)
        
        dev_dataset = BertNerDataset(self.tokenizer, self.dev_sents, self.dev_tags, self.tag2id)
        dev_dataloader = DataLoader(dev_dataset,batch_size = self.batch_size)

        for epoch in range(self.epoches):
            self.step = 0
            losses = 0.
            for idx, batch in enumerate(dataloder):
                batch = tuple(t.to(args.device) for t in batch)
                losses += self.train_step(batch, self.tag2id)
                if self.step % self.print_step == 0:
                    total_step = (len(self.train_sents) // self.batch_size + 1)
                    print("epoch {}, steps: {}/{} {:.2f}% loss: {:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0
                    break
            val_loss = self.validate(dev_dataloader, self.tag2id)
            print('epoch {}, val loss: {:.4f}'.format(epoch, val_loss))
    def train_step(self, batch, tag2id):
        self.model.train()
        self.step += 1

        outputs = self.model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],attention_mask=batch['attention_mask'])
        logits = outputs[0]
        self.optimizer.zero_grad()
        targets = batch['tag_ids']
        loss = self.loss_fn(logits, targets, tag2id)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dataloader, tag2id):
        self.model.eval()
        with torch.no_grad():
            losses = 0. 
            steps = 0
            for idx, batch in enumerate(dataloader):
                steps += 1
                batch = tuple(t.to(args.device) for t in batch)
                outputs = self.model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],attention_mask=batch['attention_mask'])
                logits = outputs[0]
                targets = batch['tag_ids']
                loss = self.loss_fn(logits, targets, tag2id)
                losses += loss.item()
                break
            val_loss = losses / steps
            if val_loss < self._best_val_loss:
                print(f'save model...')
                # TODO: implement save model
                self._best_val_loss = val_loss
            return val_loss
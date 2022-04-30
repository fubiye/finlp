import os
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig , AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from finlp.model.bert import BertSoftmaxForNerModel
from finlp.loss.util import cross_entropy

class BertNerDataset(Dataset):
    
    def __init__(self, tokenizer, sents, tags, tag2id):
        self.tokenizer = tokenizer
        self.sentences = sents
        self.tags = tags
        self.tag2id = tag2id
        self.samples = []
        self.build_dataset()
        

    def build_dataset(self):
                    
        tokenized_inputs = self.tokenizer(self.sentences, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
                
        for idx, (input_ids,token_type_ids,attention_mask) in enumerate(zip(tokenized_inputs['input_ids'],tokenized_inputs['token_type_ids'],tokenized_inputs['attention_mask'])):
            tags = self.tags[idx]
            tag_ids = [self.tag2id[tag] for tag in tags]
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_id = -1
            token_tag_ids = []
            for word_id in word_ids:
                if word_id is None:
                    token_tag_ids.append(-100)
                elif word_id == previous_word_id:
                    token_tag_ids.append(-100)
                else:
                    token_tag_ids.append(tag_ids[word_id])
            self.samples.append({
                'sample_id': idx,
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'tag_ids': torch.LongTensor(token_tag_ids)
            })

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

class BertTrainer():

    def __init__(self,train_data, dev_data, test_data,
                    word2id, tag2id):
        self.train_sents, self.train_tags = train_data
        self.dev_sents, self.dev_tags = dev_data
        self.test_sents, self.test_tags = test_data

        self.word2id = word2id
        self.tag2id = tag2id

        vocab_size = len(word2id)
        output_size = len(tag2id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("start init bert model")
        cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')
        model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
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
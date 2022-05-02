import torch

from finlp.model.rnn import BiLstm
from finlp.loss.util import cross_entropy
from finlp.metrics.entity import EntityMetrics
from finlp.metrics.tag import TagMetrics

class RnnTrainer():

    def __init__(self, train_loader, dev_loader, test_loader, tokenizer, tag2id, id2tag):
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.tag2id = tag2id
        self.id2tag = id2tag
        self.init_params()
        self.model = BiLstm(
            self.vocab_size, 
            self.embedding_dim,
            self.hidden_size,
            self.output_size 
        )
        self.model.to(self.device)
        self.loss_fn = cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def init_params(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.embedding_dim = 128
        self.hidden_size = 128
        self.vocab_size = self.tokenizer.vocab_size
        self.output_size = len(self.tag2id) - 1 # remove <pad>
        self.epoches = 5
        self.lr = 1e-3
        self.print_step = 5

    def train(self):
        self.model.train()
        for epoch in range(self.epoches):
            losses = 0.
            step = 0
            for idx, batch in enumerate(self.train_loader):
                # batch = next(iter(self.train_loader))
                step += 1
                self.model.train()
                padded_tokens = batch['padded_tokens'].to(self.device)
                seq_lengths = batch['seq_lengths']
                logits = self.model(padded_tokens, seq_lengths)
                padded_tags = batch['padded_tags'].to(self.device)

                self.optimizer.zero_grad()
                loss = self.loss_fn(logits, padded_tags, self.tag2id)
                loss.backward()
                self.optimizer.step()

                losses += loss.item()

                if step % self.print_step == 0:
                    print("training epoch {}, steps: {} loss: {:.4f}".format(
                        epoch, step, losses / self.print_step
                    ))
                    losses = 0
            val_loss = self.validate()
            print('epoch {}, val loss: {:.4f}'.format(epoch, val_loss))
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
                padded_tokens = batch['padded_tokens'].to(self.device)
                seq_lengths = batch['seq_lengths']
                logits = self.model(padded_tokens, seq_lengths)
                padded_tags = batch['padded_tags'].to(self.device)
                loss = self.loss_fn(logits, padded_tags, self.tag2id)
                losses += loss.item()

                batch_predicts = torch.argmax(logits, dim=2)
                targets.append(padded_tags)
                predicts.append(batch_predicts)

            avg_loss = losses / step
            print("{} loss: {}".format(mode, avg_loss))
            # metrics = EntityMetrics(self.tag2id, self.id2tag)
            metrics = TagMetrics(self.tag2id, self.id2tag)
            result = metrics.report(targets, predicts)
            metrics.print_result(result)
            return avg_loss
    

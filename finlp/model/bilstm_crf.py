import torch
from finlp.loss.util import cross_entropy
from finlp.model.bilstm import BiLstm
from finlp.util.seq_util import sort_by_lengths, tensorized

class BiLstmModel():

    def __init__(self, vocab_size, output_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_dim = 128
        self.hidden_size = 128

        self.model = BiLstm(vocab_size, self.embedding_dim, self.hidden_size, output_size)
        self.loss_fn = cross_entropy

        self.epoches = 30
        self.print_step = 5
        self.lr = 1e-3
        self.batch_size = 32

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, train_sents, train_tags, dev_sents, dev_tags, word2id, tag2id):
        train_sents, train_tags, _ = sort_by_lengths(train_sents, train_tags)
        dev_sents, dev_tags, _ = sort_by_lengths(dev_sents, dev_tags)

        for epoch in range(self.epoches):
            self.step = 0
            losses = 0.
            for idx in range(0, len(train_sents), self.batch_size):
                batch_sents = train_sents[idx:idx+self.batch_size]
                batch_tags = train_tags[idx:idx+self.batch_size]
                losses += self.train_step(batch_sents, batch_tags, word2id, tag2id)
                
                if self.step % self.print_step == 0:
                    total_step = (len(train_sents) // self.batch_size + 1)
                    print("epoch {}, steps: {}/{} {:.2f}% loss: {:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0

    def train_step(self, sents, tags, word2id, tag2id):
        self.model.train()
        self.step += 1

        tensorized_sents, lengths = tensorized(sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        
        targets, lengths = tensorized(tags, tag2id)
        targets = targets.to(self.device)

        logits = self.model(tensorized_sents, lengths)

        self.optimizer.zero_grad()
        loss = self.loss_fn(logits, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()



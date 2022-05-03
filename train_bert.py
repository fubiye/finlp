from finlp.transformer.trainer import SoftmaxNerTrainer
from finlp.util.init import bootstrap
if __name__ == "__main__":
    args = bootstrap()
    trainer = SoftmaxNerTrainer(args)
    trainer.main()

from finlp.transformer.trainer import SoftmaxNerTrainer
from finlp.util.init import bootstrap
if __name__ == "__main__":
    # model_name = 'bert-base-uncased'
    # dataset = 'conlldev'
    # train_file = os.path.expanduser(f"~/.cache/{dataset}/train.txt")
    # dev_file = os.path.expanduser(f"~/.cache/{dataset}/dev.txt")
    # test_file = os.path.expanduser(f"~/.cache/{dataset}/test.txt")
    # labels_file = os.path.expanduser(f"~/.cache/{dataset}/labels.txt")
    # main()
    args = bootstrap()
    trainer = SoftmaxNerTrainer(args)
    trainer.main()

import logging
import argparse
import os

logger = logging.getLogger(__name__)


class ParserInit():

    def __init__(self):
        logger.info("initializing training program...")
        self.parser = argparse.ArgumentParser()
        self.add_args()
        self.opt = self.parser.parse_args()

    def add_args(self):
        self.add_data_params(self.parser)
        self.add_training_params(self.parser)
        self.add_embedding_params(self.parser)
        self.add_hyper_parameters(self.parser)
        self.add_model_parameters(self.parser)
        self.add_transformers_parameter(self.parser)

    def add_data_params(self, parser):
        home = os.path.expanduser('~')
        cache_path = os.path.join(home, '.cache')

        parser.add_argument('--cache_dir', default=cache_path, type=str, help="the path to store data")
        glove_cache_path = os.path.join(cache_path, 'glove')
        parser.add_argument('--glove_cache', default=glove_cache_path, type=str, help="cache dir for glove")
        parser.add_argument('--dataset', default='conlldev', type=str, help="which dataset to load")
        parser.add_argument('--labels', default='labels.txt', type=str)
        parser.add_argument('--max_seq_length', default=128, type=int)
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument('--num_labels', default=9, help='total count of tags')

    def add_training_params(self, parser):
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser.add_argument('--epoches', default=10, type=int)
        parser.add_argument('--batch_size', default=16, type=int, help='batch size')
        parser.add_argument(
            "--max_steps",
            default=-1,
            type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
        )
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--logging_steps", type=str, default='0.1', help="Log every X updates steps.")
        parser.add_argument(
            "--output_dir",
            default='',
            type=str,
            required=True,
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument(
            "--evaluate_during_training",
            action="store_true",
            help="Whether to run evaluation during training at each logging step.",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--bert_lr", type=float, help="The initial learning rate for BERT.")
        parser.add_argument("--classifier_lr", type=float, help="The initial learning rate of classifier.")
        parser.add_argument("--crf_lr", type=float, help="The initial learning rate of crf")
        parser.add_argument("--adv_training", default=None, choices=['fgm', 'pgd'], help="fgm adversarial training")

        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    def add_embedding_params(self, parser):
        parser.add_argument('--pretrain', default='glove', type=str, help="(glove|....)")
        parser.add_argument('--vector_name', default='6B', type=str)
        parser.add_argument('--embedding_dim', default=100, type=int, help="embedding size")

    def add_hyper_parameters(self, parser):
        # Model Hyper Parameters
        # LSTM
        parser.add_argument('--lstm_hidden_dim', default=200, type=int, help='hidden dim for lstm')
        parser.add_argument('--lstm_bid', default=False, type=bool, help='lstm bidirectional')
        parser.add_argument('--dropout_rate', default=0.2, type=float)

    def add_model_parameters(self, parser):
        parser.add_argument('--ckpt_dir', default='checkpoints')
        parser.add_argument('--model_name', default='bert-softmax')
        parser.add_argument('--model_type', default='bert')
        parser.add_argument('--model_name_or_path', default='bert-base-uncased')
        parser.add_argument("--loss_type", default="lsr", type=str, help="The loss function to optimize.")

    def add_transformers_parameter(self, parser):
        transformers_cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')
        parser.add_argument('--transformers_cache_dir', default=transformers_cache_dir)






import logging
import torch
import random
import numpy as np
from finlp.util.args import ParserInit

logger = logging.getLogger(__name__)

def bootstrap():
    config_logging()
    logger.info("start bootstraping application")
    args = parse_arguments()
    detect_device(args)
    set_seed(args)
    bootstrap_report(args)
    return args

def parse_arguments():
    args = ParserInit().opt
    return args
def config_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

def detect_device(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    args.n_gpu = 1 if device == 'cuda' else 0

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def bootstrap_report(args):
    logger.info("*******************************")
    logger.info("DEVICE: %s, Random seed: %s", args.device, args.seed)
    logger.info("*******************************")
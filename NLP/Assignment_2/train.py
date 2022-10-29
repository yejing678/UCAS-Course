import os
import sys
import time
import random
import math
import numpy as np
import torch
import torch.nn
import logging

from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from data_utils.io_utils import *
from data_utils.vocab_utils import Vocabulary
from data_utils.prepare_data import prepare_data
from models import *
from data_utils.data import Corpus

sys.path.append(os.pardir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt, vocab_size):
        self.opt = opt
        self.vocab_size = vocab_size
        self.model = opt.model_class(opt, self.vocab_size).to(opt.device)
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params = 0
        n_nontrainable_params = 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'
                    .format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        print('resetting parameters...')
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        print("start training...")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        min_val_ppl = float("inf")
        min_val_epoch = 0

        for epoch in range(self.opt.epochs):
            logger.info('>' * 100)
            start = time.time()
            total_loss = 0.0
            step = 0
            with tqdm(train_data_loader, desc="Training epoch {}".format(epoch + 1), position=0, leave=False,
                      colour='green', ncols=100) as pbar:
                for batch in pbar:
                    self.model.train()
                    x, y = batch
                    output = self.model(x)
                    loss = criterion(output, y)
                    loss = loss.mean()
                    loss.backward()
                    total_loss += loss.item()
                    step += 1
                    # ppl stand for perplexity
                    pbar.set_postfix({'loss': total_loss / step, 'ppl': np.exp(total_loss / step),
                                      'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if step % opt.log_step == 0:
                        self.model.eval()
                        val_loss = 0.0
                        val_step = 0
                        with torch.no_grad():
                            for batch_val in val_data_loader:
                                x_val, y_val = batch_val
                                output_val = self.model(x_val)
                                loss = criterion(output_val, y_val)
                                loss = loss.mean()
                                val_loss += loss.item()
                                val_step += 1
                        average_loss = val_loss / val_step
                        val_ppl = np.exp(average_loss)
                        logger.info("\n model at step {}, nll loss:{}, ppl: {}".format(step, average_loss, val_ppl))

                        if val_ppl < min_val_ppl:
                            min_val_ppl = val_ppl
                            min_val_epoch = epoch
                            save_path = "state_dict/{0}/{1}/".format(opt.model_name, opt.seed)
                            dirname = os.path.dirname(save_path)
                            if not os.path.exists(dirname):
                                os.makedirs(dirname)

                            path = save_path + 'epoch {0}_step {1}_ppl {2}' \
                                .format(epoch, step, round(val_ppl, 4))
                            if epoch > 0:
                                # timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
                                torch.save(self.model.state_dict(), path)
                                logger.info('>> saved: {}'.format(path))

            end = time.time()
            logger.info('time: {:.4f}s'.format(end - start))
            if epoch - min_val_epoch >= opt.patience:
                logger.info('>> early stop.')
                break

        logger.info('>> min_val_acc: {:.4f}'.format(min_val_ppl))
        return path

    def _evaluate_ppl(self, criterion, data_loader):
        # switch model to evaluation mode
        # self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                output = self.model(x)
                loss = criterion(output, y)
        val_ppl = np.exp(loss)
        logger.info("\n nll loss:{}, ppl: {}".format(loss, val_ppl))
        return val_ppl

    def run(self, train_data_loader, val_data_loader, test_data_loader):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg, eps=1e-8)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        logger.info('>> best_model_path: {}'.format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path))
        test_ppl = self._evaluate_ppl(criterion, test_data_loader)
        logger.info('>> test_ppl: {:.4f}'.format(test_ppl))


def create_batch(d, batch_size, seq_len, device):
    x = []
    y = []

    x = [d[i - seq_len:i] for i in range(seq_len, len(d))]
    y = d[seq_len:]

    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def data_prepare():
    if not (os.path.isfile("./data/dict.json") and os.path.isfile("./data/train_input_ids") and os.path.isfile(
            "./data/valid_input_ids") and os.path.isfile("./data/test_input_ids")):
        my_corpus = Corpus(dic_path=None)
        my_corpus.construct_dict("./data/data.txt")
        with open("./data/data.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
        train_data, valid_test_data = train_test_split(data, test_size=0.2, random_state=42)
        valid_data, test_data = train_test_split(valid_test_data, test_size=0.5, random_state=42)
        with open("./data/train_data.txt", "w", encoding="utf-8") as f:
            f.writelines(train_data)
        with open("./data/valid_data.txt", "w", encoding="utf-8") as f:
            f.writelines(valid_data)
        with open("./data/test_data.txt", "w", encoding="utf-8") as f:
            f.writelines(test_data)

        train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = my_corpus.create_input_ids(
            "./data")
    else:
        my_corpus = Corpus(dic_path="./data/dict.json")
        train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = torch.load(
            "./data/train_input_ids_flatten"), torch.load("./data/valid_input_ids_flatten"), torch.load(
            "./data/test_input_ids_flatten")
    train_data_loader = create_batch(train_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.window_size,
                                     device=opt.device)
    val_data_loader = create_batch(valid_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.window_size,
                                   device=opt.device)
    test_data_loader = create_batch(test_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.window_size,
                                    device=opt.device)
    vocab_size = len(my_corpus.dictionary.word2idx)
    return vocab_size, train_data_loader, val_data_loader, test_data_loader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fnn', help="fnn, lstm, self_attention")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lr', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-2, type=float)
    parser.add_argument('--epochs', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--window_size', default=5, type=int)
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--max_norm', default=1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    opt = parser.parse_args()


    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'fnn': FNN,
        'lstm': LSTM,
        'transformer': Transformer
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    vocab_size, train_data_loader, val_data_loader, test_data_loader = data_prepare()
    ins = Instructor(opt, vocab_size)
    ins.run(train_data_loader, val_data_loader, test_data_loader)

    log_file = 'train:{0}-{1}-{2}-{3}-{4}-{5}.log' \
        .format(opt.model_name, opt.lr, opt.l2reg, opt.batch_size, opt.dropout, opt.seed)
    logger.addHandler(logging.FileHandler(log_file))

import os
import sys
import time
import random
import math
import numpy as np
import copy
import torch
import torch.nn
import logging
from pathlib import Path

from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from data_utils.io_utils import *
from models import *
from data_utils.data import create_batch_fnn, create_batch_rnn, data_prepare
from train_picture import plot_jasons_lineplot

sys.path.append(os.pardir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def batchify(data, bsz, device):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    data = torch.tensor(data, dtype=torch.long)
    seq_len = data.shape[0] // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


max_seq_len = 30


def get_batch(source, i):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(max_seq_len, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


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

    def _train_for_transformer(self, criterion, optimizer, train_data_loader, val_data_loader):
        print("start training...")
        update_num_list, train_loss_list, train_ppl_list, val_loss_list, val_ppl_list = [], [], [], [], []
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        min_val_ppl = float('inf')
        min_val_epoch = 0
        best_model = None
        global_step = 0

        for epoch in range(self.opt.epochs):
            start = time.time()
            total_loss = 0.0
            step = 0
            with tqdm(train_data_loader, desc="Training epoch {}".format(epoch + 1), position=0, leave=False,
                      colour='green', ncols=100) as pbar:
                for batch in pbar:
                    self.model.train()
                    x, y = batch
                    mask = generate_square_subsequent_mask(self.opt.seq_len).to(self.opt.device)
                    output = self.model(x.view(x.shape[1], x.shape[0]), mask)
                    loss = criterion(output.view(-1, vocab_size), y.view(-1))
                    loss = loss.mean()
                    loss.backward()
                    total_loss += loss.item()
                    step += 1
                    global_step += 1
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
                                mask_val = generate_square_subsequent_mask(self.opt.seq_len).to(self.opt.device)
                                output_val = self.model(x_val.view(x_val.shape[1], x_val.shape[0]), mask_val)
                                loss = criterion(output_val.view(-1,vocab_size), y_val.view(-1))
                                val_loss += loss.item()
                                val_step += 1
                        average_loss = val_loss / val_step
                        val_ppl = math.exp(average_loss)
                        print('\n')
                        logger.info("| epoch {:3d} | step {:6d} | nnl loss {:5.3f} | val_ppl {:8.3f} "
                                    .format(epoch + 1, step, average_loss, val_ppl))
                        update_num_list.append(global_step)
                        train_loss_list.append(total_loss / step)
                        train_ppl_list.append(np.exp(total_loss / step))
                        val_ppl_list.append(val_ppl)
                        val_loss_list.append(average_loss)
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
                                best_model = copy.deepcopy(self.model)
                                # timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
                                torch.save(self.model.state_dict(), path)
                                logger.info('>> saved: {}'.format(path))

            end = time.time()
            logger.info('-' * 120)
            logger.info(
                '| end of epoch {:3d} | time: {:.4f}s | valid loss {:.4f} | valid ppl {:.4f}'.
                format(epoch + 1, end - start, average_loss, val_ppl))
            logger.info('-' * 120)

            if epoch - min_val_epoch >= opt.patience:
                logger.info('>> early stop.')
                break

            # draw picture
            save_picture_path = "plots/model-{}/batch_size-{}/window_size-{}/seed-{}". \
                format(self.opt.model_name, self.opt.batch_size, self.opt.window_size, self.opt.seed)
            picture_dirname = os.path.dirname(save_picture_path)
            if not os.path.exists(picture_dirname):
                os.makedirs(picture_dirname)
            train_loss_path = save_picture_path + 'train_loss.jpg'
            train_ppl_path = save_picture_path + 'train_ppl.jpg'
            val_loss_path = save_picture_path + 'val_loss.jpg'
            val_ppl_path = save_picture_path + 'val_ppl.jpg'

            plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                 f"{self.opt.model_name}  min_train_loss={min(train_loss_list):.3f}",
                                 train_loss_path)
            plot_jasons_lineplot(update_num_list, val_loss_list, 'updates', 'validation loss',
                                 f"{self.opt.model_name}  min_val_loss={min(val_loss_list):.3f}",
                                 val_loss_path)
            plot_jasons_lineplot(update_num_list, train_ppl_list, 'updates', 'train perplexity',
                                 f"{self.opt.model_name}  min_train_ppl={min(train_ppl_list):.3f}",
                                 train_ppl_path)
            plot_jasons_lineplot(update_num_list, train_ppl_list, 'updates', 'val perplexity',
                                 f"{self.opt.model_name}  min_val_ppl={min(val_ppl_list):.3f}",
                                 val_ppl_path)

        logger.info('>> min_val_ppl: {:.4f}'.format(min_val_ppl))
        return path, best_model

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        print("start training...")
        update_num_list, train_loss_list, train_ppl_list, val_loss_list, val_ppl_list = [], [], [], [], []
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        min_val_ppl = float('inf')
        min_val_epoch = 0
        best_model = None
        global_step = 0

        for epoch in range(self.opt.epochs):
            start = time.time()
            total_loss = 0.0
            step = 0
            with tqdm(train_data_loader, desc="Training epoch {}".format(epoch + 1), position=0, leave=False,
                      colour='green', ncols=100) as pbar:
                for batch in pbar:
                    self.model.train()
                    x, y = batch
                    output = self.model(x)
                    loss = criterion(output, y.view(-1))
                    loss.backward()
                    total_loss += loss.item()
                    step += 1
                    global_step += 1
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
                                loss = criterion(output_val, y_val.view(-1))
                                val_loss += loss.item()
                                val_step += 1
                        average_loss = val_loss / val_step
                        val_ppl = math.exp(average_loss)
                        print('\n')
                        logger.info("| epoch {:3d} | step {:6d} | nnl loss {:5.3f} | val_ppl {:8.3f} "
                                    .format(epoch + 1, step, average_loss, val_ppl))
                        update_num_list.append(global_step)
                        train_loss_list.append(total_loss / step)
                        train_ppl_list.append(np.exp(total_loss / step))
                        val_ppl_list.append(val_ppl)
                        val_loss_list.append(average_loss)
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
                                best_model = copy.deepcopy(self.model)
                                # timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
                                torch.save(self.model.state_dict(), path)
                                logger.info('>> saved: {}'.format(path))

            end = time.time()
            logger.info('-' * 120)
            logger.info(
                '| end of epoch {:3d} | time: {:.4f}s | valid loss {:.4f} | valid ppl {:.4f}'.
                format(epoch + 1, end - start, average_loss, val_ppl))
            logger.info('-' * 120)

            if epoch - min_val_epoch >= opt.patience:
                logger.info('>> early stop.')
                break

            # draw picture
            save_picture_path = "plots/model-{}/batch_size-{}/window_size-{}/seed-{}". \
                format(self.opt.model_name, self.opt.batch_size, self.opt.window_size, self.opt.seed)
            picture_dirname = os.path.dirname(save_picture_path)
            if not os.path.exists(picture_dirname):
                os.makedirs(picture_dirname)
            train_loss_path = save_picture_path + 'train_loss.jpg'
            train_ppl_path = save_picture_path + 'train_ppl.jpg'
            val_loss_path = save_picture_path + 'val_loss.jpg'
            val_ppl_path = save_picture_path + 'val_ppl.jpg'

            plot_jasons_lineplot(update_num_list, train_loss_list, 'updates', 'training loss',
                                 f"{self.opt.model_name}  min_train_loss={min(train_loss_list):.3f}",
                                 train_loss_path)
            plot_jasons_lineplot(update_num_list, val_loss_list, 'updates', 'validation loss',
                                 f"{self.opt.model_name}  min_val_loss={min(val_loss_list):.3f}",
                                 val_loss_path)
            plot_jasons_lineplot(update_num_list, train_ppl_list, 'updates', 'train perplexity',
                                 f"{self.opt.model_name}  min_train_ppl={min(train_ppl_list):.3f}",
                                 train_ppl_path)
            plot_jasons_lineplot(update_num_list, train_ppl_list, 'updates', 'val perplexity',
                                 f"{self.opt.model_name}  min_val_ppl={min(val_ppl_list):.3f}",
                                 val_ppl_path)

        logger.info('>> min_val_ppl: {:.4f}'.format(min_val_ppl))
        return path, best_model

    def _evaluate_ppl_for_transformer(self, criterion, data_loader):
        # switch model to evaluation mode
        self.model.eval()
        total_loss = 0.0
        step = 0
        with torch.no_grad():
            for batch_test in data_loader:
                step += 1
                x_test, y_test = batch_test
                mask_test = generate_square_subsequent_mask(self.opt.seq_len).to(self.opt.device)
                output_test = self.model(x_test.view(x_test.shape[1], x_test.shape[0], mask_test))
                loss = criterion(output_test, y_test.view(-1))
                loss = loss.mean()
                total_loss += loss.item()
            average_loss = round(total_loss / step, 3)
            test_ppl = round(np.exp(average_loss), 3)
        return average_loss, test_ppl

    def _evaluate_ppl(self, criterion, data_loader):
        # switch model to evaluation mode
        self.model.eval()
        total_loss = 0.0
        step = 0
        with torch.no_grad():
            for batch_test in data_loader:
                step += 1
                x_test, y_test = batch_test
                output_test = self.model(x_test)
                loss = criterion(output_test, y_test.view(-1))
                loss.mean()
                total_loss += loss.item()
            average_loss = round(total_loss / step, 3)
            test_ppl = round(np.exp(average_loss), 3)
        return average_loss, test_ppl

    def run(self, train_data_loader, val_data_loader, test_data_loader):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg, eps=1e-8)

        self._reset_params()
        if self.opt.model_name == 'transformer':
            best_model_path, best_model = self._train_for_transformer(criterion, optimizer, train_data_loader,
                                                                      val_data_loader)
            logger.info('>> best_model_path: {}'.format(best_model_path))
            print('loading best model...')
            self.model.load_state_dict(torch.load(best_model_path))
            loss, test_ppl = self._evaluate_ppl_for_transformer(criterion, test_data_loader)
            logger.info("test nll loss:{:.3f}, ppl: {:.3f}".format(loss, test_ppl))
        else:
            best_model_path, best_model = self._train(criterion, optimizer, train_data_loader, val_data_loader)
            # best_model_path = "/home/jye/UCAS_Course/NLP/state_dict/lstm/42/epoch 4_step 6000_ppl 74.4185"
            logger.info('>> best_model_path: {}'.format(best_model_path))
            print('loading best model...')
            self.model.load_state_dict(torch.load(best_model_path))
            loss, test_ppl = self._evaluate_ppl(criterion, test_data_loader)
            logger.info("test nll loss:{:.3f}, ppl: {:.3f}".format(loss, test_ppl))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='transformer', help="fnn, lstm, transformer", type=str)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-2, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--log_step', default=1000, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--feedforward_dim', default=256, type=int)
    parser.add_argument('--window_size', default=5, type=int)
    parser.add_argument('--seq_len', default=5, type=int)
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--max_norm', default=1, type=int)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--device', default='cuda:1', type=str, help='e.g. cuda:0')
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
        'adamw': torch.optim.AdamW
    }

    opt.model_class = model_classes[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    log_file = 'train:{0}-{1}-{2}-{3}-{4}-{5}-{6}.log' \
        .format(opt.model_name, opt.lr, opt.l2reg, opt.batch_size, opt.window_size, opt.dropout, opt.seed)
    logger.addHandler(logging.FileHandler(log_file))

    vocab_size, train_input_ids_flatten, valid_input_ids_flatten, test_input_ids_flatten = data_prepare(opt)
    if opt.model_name == 'fnn':
        train_data_loader = create_batch_fnn(train_input_ids_flatten, batch_size=opt.batch_size,
                                             seq_len=opt.window_size,
                                             device=opt.device)
        val_data_loader = create_batch_fnn(valid_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.window_size,
                                           device=opt.device)
        test_data_loader = create_batch_fnn(test_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.window_size,
                                            device=opt.device)
        ins = Instructor(opt, vocab_size)
        ins.run(train_data_loader, val_data_loader, test_data_loader)
    elif opt.model_name == 'lstm' or opt.model_name == 'transformer':
        train_data_loader = create_batch_rnn(train_input_ids_flatten, batch_size=opt.batch_size,
                                             seq_len=opt.seq_len, device=opt.device)
        val_data_loader = create_batch_rnn(valid_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.seq_len,
                                           device=opt.device)
        test_data_loader = create_batch_rnn(test_input_ids_flatten, batch_size=opt.batch_size, seq_len=opt.seq_len,
                                            device=opt.device)
        ins = Instructor(opt, vocab_size)
        ins.run(train_data_loader, val_data_loader, test_data_loader)

"""
from https://github.com/ibalazevic/TuckER
"""
import time
import argparse
import datetime
import os
from collections import defaultdict

from load_data import Data
from model import TuckER
from interpretability_evaluation import getDistRatio, pick_top_k
from gRDA import gRDA_momentum, gRDAAdam

# Init wandb
import wandb

import numpy as np
import torch
import torch.nn.functional as F


class Experiment:
    def __init__(self,
                 learning_rate=0.0005,
                 ent_vec_dim=200,
                 rel_vec_dim=200,
                 num_iterations=500,
                 batch_size=128,
                 decay_rate=0.,
                 cuda=False,
                 input_dropout=0.3,
                 hidden_dropout1=0.4,
                 hidden_dropout2=0.5,
                 label_smoothing=0.,
                 loss='CE',
                 optimizer='RDA',
                 reg=10e-08,
                 mu=0.5,
                 c=0.0005):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.loss = loss
        self.optimizer = optimizer
        self.reg = reg
        self.mu = mu
        self.c = c
        self.kwargs = {
            "input_dropout": input_dropout,
            "hidden_dropout1": hidden_dropout1,
            "hidden_dropout2": hidden_dropout2,
            "loss": loss,
        }

    def adjust_learning_rate(self, epoch, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
        lr = args.lr * (0.1**(epoch // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data, it):
        best_mrr = 0
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, eval_targets = self.get_batch(er_vocab, test_data_idxs,
                                                      i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx, eval_targets)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        mrr = np.mean(1. / np.array(ranks))
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(mrr))

        wandb.log({'val_mean_rank': np.mean(ranks)}, step=it)
        wandb.log({'val_mrr': mrr}, step=it)

        if (mrr > best_mrr):
            wandb.run.summary["best_mrr"] = mrr
            best_mrr = mrr

    def train_and_eval(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.idxs_entity = {i: d.entities[i] for i in range(len(d.entities))}
        self.relation_idxs = {
            d.relations[i]: i
            for i in range(len(d.relations))
        }

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)

        wandb.watch(model, log="all")

        if self.cuda:
            model.cuda()
        model.init()

        if self.optimizer == 'Adam':
            opt = torch.optim.Adam(model.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=0.0)

        elif self.optimizer == 'SGD':
            opt = torch.optim.SGD(model.parameters(),
                                  lr=self.learning_rate,
                                  momentum=0.9)

        elif self.optimizer == 'RDA':
            print('Using RDA!')
            opt = gRDA_momentum(model.parameters(),
                                lr=self.learning_rate,
                                c=self.c,
                                mu=self.mu,
                                momentum=0.9,
                                reg='l1')

        elif self.optimizer == 'RDA_adam':
            print('Using RDA_Adam!')
            opt = gRDAAdam(model.parameters(),
                           lr=self.learning_rate,
                           c=self.c,
                           mu=self.mu,
                           reg='l1')

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        start_time = time.time()
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)

            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs,
                                                     j)

                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx, targets)

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) *
                               targets) + (1.0 / targets.size(1))

                if self.kwargs['loss'] == 'CE':
                    loss = model.klloss(
                        F.log_softmax(predictions, dim=1),
                        F.normalize(targets.float(), p=1, dim=1))
                else:
                    loss = model.loss(predictions, targets)

                loss.backward()
                opt.step()
                opt.zero_grad()

                losses.append(loss.item())

            print('--' * 30)
            print('Iteration {}'.format(it))
            print('Total Time (h:m:s): {:2}'.format(
                str(datetime.timedelta(seconds=int(time.time() -
                                                   start_time)))))
            print('Iteration Time (h:m:s): {:2}'.format(
                str(datetime.timedelta(seconds=int(time.time() -
                                                   start_train)))))
            print('Mean loss: {}'.format(np.mean(losses)))
            wandb.log({'loss': np.mean(losses)}, step=it)
            model.eval()
            with torch.no_grad():

                print('++' * 20)
                print("Validation:")
                self.evaluate(model, d.valid_data, it)
                if not it % 5:
                    print('##' * 20)
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data, it)
                    print('Test Time (h:m:s): {:2}'.format(
                        str(
                            datetime.timedelta(seconds=int(time.time() -
                                                           start_test)))))

                if not it % 100:
                    pick_top_k(model.E, d.entity_ids_to_readable,
                               self.idxs_entity)

                print('Sparsity (entity embeddings): {}'.format(
                    model.count_zero_weights_ent()))
                wandb.log(
                    {'ent_spars': np.mean(model.count_zero_weights_ent())},
                    step=it)
                print('Sparsity (relation embeddings): {}'.format(
                    model.count_zero_weights_rel()))
                wandb.log(
                    {'rel_spars': np.mean(model.count_zero_weights_rel())},
                    step=it)
                print('Sparsity (core tensor): {}'.format(
                    model.count_zero_weights_W()))

                print('Negativity (entity embeddings): {}'.format(
                    model.count_negative_weights_ent()))
                print('Negativity (relation embeddings): {}'.format(
                    model.count_negative_weights_rel()))
                print('Negativity (core tensor): {}'.format(
                    model.count_negative_weights_W()))

                e_dr = getDistRatio(model.E)
                r_dr = getDistRatio(model.R)

                print('DistRatio (entity embeddings): {}'.format(e_dr))
                wandb.log({'ent_distratio': e_dr}, step=it)
                print('DistRatio (relation embeddings): {}'.format(r_dr))
                wandb.log({'rel_distratio': r_dr}, step=it)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="FB15k-237",
        nargs="?",
        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--loss",
                        type=str,
                        default="BCE",
                        nargs="?",
                        help="Which loss to use: BCE or CE.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="RDA",
                        nargs="?",
                        help="Which optimizer to use: Adam, SGD or RDA.")
    parser.add_argument("--num_iterations",
                        type=int,
                        default=2400,
                        nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0005,
                        nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr",
                        type=float,
                        default=0.1,
                        nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim",
                        type=int,
                        default=200,
                        nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim",
                        type=int,
                        default=200,
                        nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda",
                        type=bool,
                        default=True,
                        nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout",
                        type=float,
                        default=0.3,
                        nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1",
                        type=float,
                        default=0.4,
                        nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2",
                        type=float,
                        default=0.5,
                        nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.1,
                        nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--mu",
                        type=float,
                        default=0.5,
                        nargs="?",
                        help="RDA mu")
    parser.add_argument("--c",
                        type=float,
                        default=0.00005,
                        nargs="?",
                        help="RDA c")
    parser.add_argument("--nowandb",
                        dest='nowandb',
                        action='store_true',
                        default=False,
                        help="Log wandb.")

    args = parser.parse_args()

    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    if args.nowandb:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project="nnskge")
    wandb.config.update(args)  # adds all of the arguments as config variables

    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=args.num_iterations,
                            batch_size=args.batch_size,
                            learning_rate=args.lr,
                            decay_rate=args.dr,
                            ent_vec_dim=args.edim,
                            rel_vec_dim=args.rdim,
                            cuda=args.cuda,
                            input_dropout=args.input_dropout,
                            hidden_dropout1=args.hidden_dropout1,
                            hidden_dropout2=args.hidden_dropout2,
                            label_smoothing=args.label_smoothing,
                            optimizer=args.optimizer,
                            loss=args.loss,
                            mu=args.mu,
                            c=args.c)
    experiment.train_and_eval()
    print('#' * 40)
    print("Finished!")

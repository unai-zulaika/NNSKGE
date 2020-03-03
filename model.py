"""
from https://github.com/ibalazevic/TuckER
"""

import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                         dtype=torch.float,
                         device="cuda",
                         requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        if kwargs["loss"] == 'BCE':
            self.loss = torch.nn.BCELoss()
        elif kwargs["loss"] == 'CE':
            self.loss = torch.nn.CrossEntropyLoss()
            self._klloss = torch.nn.KLDivLoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx, y=0):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def count_zero_weights_ent(self):
        zeros = 0
        p = 0
        for param in self.E.weight:
            p += param.size()[0]
            zeros += param.numel() - param.nonzero().size(0)
        return (zeros / p)

    def count_zero_weights_rel(self):
        zeros = 0
        p = 0
        for param in self.R.weight:
            p += param.size()[0]
            zeros += param.numel() - param.nonzero().size(0)
        return (zeros / p)

    def count_zero_weights_W(self):
        zeros = 0
        p = 0
        for param in self.W:
            p += param.size()[0]
            zeros += param.numel() - param.nonzero().size(0)
        return (zeros / p)

    def count_negative_weights_ent(self):
        neg = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += param.size()[0]
                neg += torch.sum((param < 0)).data.item()
        return (neg / p)

    def count_negative_weights_rel(self):
        neg = 0
        p = 0
        for param in self.R.weight:
            if param is not None:
                p += param.size()[0]
                neg += torch.sum((param < 0)).data.item()
        return (neg / p)

    def count_negative_weights_W(self):
        neg = 0
        p = 0
        for param in self.W:
            if param is not None:
                p += param.size()[0]
                neg += torch.sum((param < 0)).data.item()
        return (neg / p)

"""
from https://github.com/ibalazevic/TuckER
"""

import numpy as np
import torch
from torch.nn.init import xavier_normal_, uniform_, eye_, sparse_
from proximalGradient import l1

class NNSKGE(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(NNSKGE, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        if kwargs["loss"] == 'BCE':
            self.loss = torch.nn.BCELoss()
        elif kwargs["loss"]== 'CE':
            self.loss = torch.nn.CrossEntropyLoss()

        self.prox_lr = kwargs["prox_lr"]
        self.prox_reg= kwargs["prox_reg"]

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        
    def init(self):
        # uniform_(self.E.weight.data, a=0.0, b=1.0)
        # uniform_(self.R.weight.data, a=0.0, b=1.0)
        sparse_(self.E.weight.data, sparsity=0.5)
        sparse_(self.R.weight.data, sparsity=0.5)
  
    def forward(self, e1_idx, r_idx, y):
        
        e1 = self.E(e1_idx)
    
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        # x = e1.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))

        pred = torch.sigmoid(x)
        return pred

    def regularize(self):
        self.E.weight.data = torch.clamp(self.E.weight, min=0)
        self.R.weight.data = torch.clamp(self.R.weight, min=0)
        # self.W.data = torch.clamp(self.W, min=0)
        # self.E.weight.data = torch.clamp(self.E.weight, min=0, max=1)
        # self.R.weight.data = torch.clamp(self.R.weight, min=0, max=1)
        # self.W.data = torch.clamp(self.W, min=0, max=1)
    
    def sparsity(self,):
        reg = torch.norm(self.E.weight, p=1) + torch.norm(self.R.weight, p=1)  # + torch.norm(self.W, p=1)
        return reg

    def proximal(self,):
        l1(self.E.weight, reg=self.prox_reg, lr=self.prox_lr)
        l1(self.R.weight, reg=self.prox_reg, lr=self.prox_lr)

    def countZeroWeightsEnt(self):
        zeros = 0
        p = 0
        for param in self.E.weight:
            p += param.size()[0]
            zeros += param.numel() - param.nonzero().size(0)
        return (zeros/p)

    def countZeroWeightsRel(self):
        zeros = 0
        p = 0
        for param in self.R.weight:
            p += param.size()[0]
            zeros += param.numel() - param.nonzero().size(0)
        return (zeros/p)

    def countNegativeWeights(self):
        neg = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += param.size()[0]
                neg += torch.sum((param < 0)).data.item()
        return (neg/p)


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        if kwargs["loss"] == 'BCE':
            self.loss = torch.nn.BCELoss()
        elif kwargs["loss"]== 'CE':
            self.loss = torch.nn.CrossEntropyLoss()

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
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
    
    def regularize(self):
        self.E.weight.data = torch.clamp(self.E.weight, min=0, max=1)
        self.R.weight.data = torch.clamp(self.R.weight, min=0, max=1)
        self.W.data = torch.clamp(self.W, min=0, max=1)

    def countZeroWeights(self):
        zeros = 0
        p = 0
        for param in self.E.weight:
            p += param.size()[0]
            zeros += param.numel() - param.nonzero().size(0)
        return (zeros/p)

    def countNegativeWeights(self):
        neg = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += param.size()[0]
                neg += torch.sum((param < 0)).data.item()
        return (neg/p)

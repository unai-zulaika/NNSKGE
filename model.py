"""
from https://github.com/ibalazevic/TuckER
"""

import numpy as np
import torch
from torch.nn.init import xavier_normal_, uniform_, eye_

class NNSKGE(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(NNSKGE, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(0, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        self.sparsity_hp = 0.5
        
    def init(self):
        # xavier_normal_(self.E.weight.data)
        # xavier_normal_(self.R.weight.data)
        uniform_(self.E.weight.data, a=0.0, b=1.0)
        uniform_(self.R.weight.data, a=0.0, b=1.0)
        # eye_(self.W.data)


    def forward(self, e1_idx, r_idx, y):
        
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
    
    def sparsity(self,):
        reg = torch.norm(self.E.weight, p=1) + torch.norm(self.R.weight, p=1) + torch.norm(self.W, p=1)
        return (self.sparsity_hp * reg) 

    def countZeroWeights(self):
        zeros = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += 1
                zeros += torch.sum((param == 0).int()).data.item()
        return int(zeros/p)

    def countNegativeWeights(self):
        neg = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += 1
                neg += torch.sum((param < 0).int()).data.item()
        return int(neg/p)

    #TODO: add Frobenius norm to embeddings


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
        self.loss = torch.nn.BCELoss()

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

    def sparsity(self,):
        return 0
    
    def regularize(self):
        i=0

    def countZeroWeights(self):
        zeros = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += 1
                zeros += torch.sum((param == 0).int()).data.item()
        return int(zeros/p)

    def countNegativeWeights(self):
        neg = 0
        p = 0
        for param in self.E.weight:
            if param is not None:
                p += 1
                neg += torch.sum((param < 0).int()).data.item()
        return int(neg/p)
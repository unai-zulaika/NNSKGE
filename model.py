import numpy as np
import torch
from torch.nn.init import xavier_normal_


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
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

        self.sparsity_hp = 1.0
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    # def forward(self, e1_idx, r_idx):
    #     e1 = self.E(e1_idx)
    #     x = self.bn0(e1)
    #     x = self.input_dropout(x)
    #     x = x.view(-1, 1, e1.size(1))

    #     r = self.R(r_idx)
    #     W_mat = torch.mm(r, self.W.view(r.size(1), -1))
    #     W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
    #     W_mat = self.hidden_dropout1(W_mat)

    #     x = torch.bmm(x, W_mat) 
    #     x = x.view(-1, e1.size(1))      
    #     x = self.bn1(x)
    #     x = self.hidden_dropout2(x)
    #     x = torch.mm(x, self.E.weight.transpose(1,0))
    #     pred = torch.sigmoid(x)
    #     return pred

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

        x = torch.norm((x - y))
        x = x**2/2
        x += self.sparsity(e1_idx, r_idx)

        pred = torch.sigmoid(x)
        return pred

    
    def sparsity(self, e1_idx, r_idx):
        E = torch.norm(self.E(e1_idx), p=1)
        R = torch.norm(self.R(r_idx), p=1)
        x = torch.sum(E + R) + torch.norm(self.W, p=1)

        return x
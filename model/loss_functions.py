from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM

        return score
    
    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        # margin from 1.0 
        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
        score = self.get_score(queries, items)      # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = items[indices[:, 0]]  # TxC
        neg = items[indices[:, 1]]  # TxC
        anc = queries              # TxC

        spread_loss = loss(anc, pos, neg)

        if self.reduce:
            return spread_loss
        
        spread_loss = spread_loss.contiguous().view(batch_size, -1)       # N x L
        
        return spread_loss     # N x L

class GatheringLoss(nn.Module):
    def __init__(self, reduce=True):
        super(GatheringLoss, self).__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM

        return score
    
    def forward(self, queries, items):
        '''
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        loss_mse = torch.nn.MSELoss(reduce=self.reduce)

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
        score = self.get_score(queries, items)      # TxM

        _, indices = torch.topk(score, 1, dim=1)

        gathering_loss = loss_mse(queries, items[indices].squeeze(1))

        if self.reduce:
            return gathering_loss
        
        gathering_loss = torch.sum(gathering_loss, dim=-1)  # T
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)   # N x L

        return gathering_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        '''
        x (attn_weights) : TxM
        '''
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss


class NearestSim(nn.Module):
    def __init__(self):
        super(NearestSim, self).__init__()
        
    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = F.linear(query, key)   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM

        return score
    
    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
        score = self.get_score(queries, items)      # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)

        # 1st and 2nd nearest items (l2 normalized)
        pos = F.normalize(items[indices[:, 0]], p=2, dim=-1)  # TxC
        anc = F.normalize(queries, p=2, dim=-1)               # TxC

        similarity = -1 * torch.sum(pos * anc, dim=-1)         # T
        similarity = similarity.contiguous().view(batch_size, -1)   # N x L
        
        return similarity     # N x L
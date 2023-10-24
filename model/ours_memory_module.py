from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans


class MemoryModule(nn.Module):
    def __init__(self, n_memory, fea_dim, shrink_thres=0.0025, device=None, memory_init_embedding=None, phase_type=None, dataset_name=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim  # C(=d_model)
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        self.memory_init_embedding = memory_init_embedding
        
        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)
        
        # mem (memory items) : M x C
        # first train -> memory_initial : False / memory_init_embedding : None
        # second_train -> memory_initial : False / memory_init_embedding : kmeans item
        # test -> memory_initial: False / memory_init_embedding : vectors from second train phase
        if self.memory_init_embedding == None:
            if self.phase_type =='test':
                # test
                # before
                # self.memory_init_embedding = torch.load('./memory_item/SMD_memory_item.pth')
                # print('loading memory item vectors trained from kmeans (for test phase)')
                # self.mem = self.memory_init_embedding
                # after
                load_path = f'./memory_item/{dataset_name}_memory_item.pth'
                self.mem = torch.load(load_path)
                print(load_path)
                print('loading memory item vectors trained from kmeans (for test phase)')

            else:
                # first train
                print('loading memory item with random initilzation (for first train phase)')

                self.mem = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1)
        else:
            # second train 
            if self.phase_type == 'second_train':
                print('second training (for second train phase)')
                # before
                # self.memory_init_embedding = memory_init_embedding
                # self.mem = self.memory_init_embedding
                # after
                self.mem = memory_init_embedding
            
    # relu based hard shrinkage function, only works for positive values
    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        
        return output
    
    def get_attn_score(self, query, key):
        '''
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C
        '''
        attn = torch.matmul(query, torch.t(key.cuda()))    # (TxC) x (CxM) -> TxM
        attn = F.softmax(attn, dim=-1)

        if (self.shrink_thres > 0):
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            # re-normalize
            attn = F.normalize(attn, p=1, dim=1)
        
        return attn
    
    def read(self, query):
        '''
        query (initial features) : (NxL) x C or N x C -> T x C
        read memory items and get new robust features, 
        while memory items(cluster centers) being fixed 
        '''
        self.mem = self.mem.cuda()
        attn = self.get_attn_score(query, self.mem.detach())  # T x M
        add_memory = torch.matmul(attn, self.mem.detach())    # T x C

        # add_memory = F.normalize(add_memory, dim=1)
        read_query = torch.cat((query, add_memory), dim=1)  # T x 2C

        return {'output': read_query, 'attn': attn}

    def update(self, query):
        '''
        Update memory items(cluster centers)
        Fix Encoder parameters (detach)
        query (encoder output features) : (NxL) x C or N x C -> T x C
        '''
        self.mem = self.mem.cuda()
        attn = self.get_attn_score(self.mem, query.detach())  # M x T
        add_mem = torch.matmul(attn, query.detach())   # M x C

        # update gate : M x C
        update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem)) # M x C
        self.mem = (1 - update_gate)*self.mem + update_gate*add_mem
        # self.mem = F.noramlize(self.mem + add_mem, dim=1)   # M x C

    def forward(self, query):
        '''
        query (encoder output features) : N x L x C or N x C
        '''
        s = query.data.shape
        l = len(s)

        query = query.contiguous()
        query = query.view(-1, s[-1])  # N x L x C or N x C -> T x C

        # Normalized encoder output features
        # query = F.normalize(query, dim=1)
        
        # update memory items(cluster centers), while encoder parameters being fixed
        if self.phase_type != 'test':
            self.update(query)
        
        # get new robust features, while memory items(cluster centers) being fixed
        outs = self.read(query)
        
        read_query, attn = outs['output'], outs['attn']
        
        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2*s[2])
            attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError('Wrong input dimension')
        '''
        output : N x L x 2C or N x 2C
        attn : N x L x M or N x M
        '''
        return {'output': read_query, 'attn': attn, 'memory_init_embedding':self.mem}
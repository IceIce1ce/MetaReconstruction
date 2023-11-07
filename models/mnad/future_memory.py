import torch
import torch.nn as nn
from torch.nn import functional as F

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim,  temp_update, temp_gather):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
    
    def get_update_query(self, mem, max_indices, score, query, train):
        m, d = mem.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0
            return query_update
        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0
            return query_update

    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()
        score = torch.matmul(query, torch.t(mem))
        score = score.view(bs * h * w, m)
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)
        return score_query, score_memory
    
    def forward(self, query, keys, train=True):
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)
        if train:
            separateness_loss, compactness_loss = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = self.update(query, keys, train)
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:
            compactness_loss, query_re, top1_keys, keys_ind = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = keys
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, query_re, top1_keys, keys_ind, compactness_loss
    
    def update(self, query, keys, train):
        batch_size, h, w, dims = query.size()
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        if train:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        else:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        return updated_memory.detach()
        
    def gather_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()
        # Feature separateness loss + feature compactness loss
        if train:
            loss = torch.nn.TripletMarginLoss(margin=1.0)
            loss_mse = torch.nn.MSELoss()
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size * h * w, dims)
            _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)
            pos = keys[gathering_indices[:, 0]]
            neg = keys[gathering_indices[:, 1]]
            top1_loss = loss_mse(query_reshape, pos.detach()) # gathering_loss
            gathering_loss = loss(query_reshape, pos.detach(), neg.detach()) # spreading_loss
            return gathering_loss, top1_loss
        # Feature compactness loss
        else:
            loss_mse = torch.nn.MSELoss()
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size * h * w, dims)
            _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
            gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
            return gathering_loss, query_reshape, keys[gathering_indices].squeeze(1).detach(), gathering_indices[:, 0]
    
    def read(self, query, keys):
        batch_size, h, w, dims = query.size()
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        concat_memory = torch.matmul(softmax_score_memory.detach(), keys)
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0, 3, 1, 2)
        return updated_query, softmax_score_query, softmax_score_memory
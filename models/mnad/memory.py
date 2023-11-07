import torch
import torch.nn as nn
from torch.nn import functional as F

# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def calc_euclidean(self, x1, x2):
#         return (x1 - x2).pow(2).sum(1)
#
#     def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
#         distance_positive = self.calc_euclidean(anchor, positive)
#         distance_negative = self.calc_euclidean(anchor, negative)
#         losses = torch.relu(distance_positive - distance_negative + self.margin)
#
#         return losses.mean()

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather

    def get_update_query(self, mem, max_indices, score, query, train):
        m, d = mem.size()  # [10, 512]
        if train:
            query_update = torch.zeros((m, d)).cuda()  # [10, 512]
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)  # right side Eq 3 and Eq 5 -> [512]
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

    # Eq 1 and Eq 4
    def get_score(self, mem, query):
        # keys -> [10, 512], query -> [4, 32, 32, 512]
        bs, h, w, d = query.size()
        m, d = mem.size()
        score = torch.matmul(query, torch.t(mem))  # [4, 32, 32, 10]
        score = score.view(bs * h * w, m)  # [4096, 10]
        score_query = F.softmax(score, dim=0)  # [4096, 10]
        score_memory = F.softmax(score, dim=1)  # [4096, 10]
        return score_query, score_memory

    def forward(self, query, keys, train=True):
        # query -> [4, 512, 32, 32] ,keys -> [mem_dim, mem_item_dim] -> [10, 512]
        query = F.normalize(query, dim=1)  # [4, 512, 32, 32]
        query = query.permute(0, 2, 3, 1)  # [4, 32, 32, 512]
        if train:
            gathering_loss = self.gather_loss(query, keys)
            spreading_loss = self.spread_loss(query, keys)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = self.update(query, keys, train)
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        else:
            gathering_loss = self.gather_loss(query, keys)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = keys
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss

    # Update module
    def update(self, query, keys, train):
        batch_size, h, w, dims = query.size()  # [4, 32, 32, 512]
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)  # [4096, 10] , [4096, 10]
        query_reshape = query.contiguous().view(batch_size * h * w, dims)  # [4096, 512]
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  # [4096, 1]
        if train:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_query, query_reshape, train)  # [10, 512]
            updated_memory = F.normalize(query_update + keys, dim=1)  # Eq 3
        else:
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        return updated_memory.detach()  # [10, 512]

    # Feature separateness loss
    def spread_loss(self, query, keys):
        batch_size, h, w, dims = query.size()  # [4, 32, 32, 512]
        loss = torch.nn.TripletMarginLoss(margin=1.0)  # alpha = 1 in Eq 12
        # calc top 2 query from Eq 1, 4 and 12
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)  # [4096, 10] , [4096, 10]
        query_reshape = query.contiguous().view(batch_size * h * w, dims)  # [4096, 512]
        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)  # [4096, 2]
        pos = keys[gathering_indices[:, 0]]  # 1st nearest item
        neg = keys[gathering_indices[:, 1]]  # 2nd nearest item
        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())  # [4096, 512], [4096, 512], [4096, 512]
        return spreading_loss

    # Feature compactness loss
    def gather_loss(self, query, keys):
        batch_size, h, w, dims = query.size()  # [4, 32, 32, 512]
        loss_mse = torch.nn.MSELoss()
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)  # [4096, 10] , [4096, 10]
        query_reshape = query.contiguous().view(batch_size * h * w, dims)  # [4096, 512]
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  # [4096, 2]
        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())  # [4096, 512], [4096, 512]
        return gathering_loss

    # Read module
    def read(self, query, keys):
        batch_size, h, w, dims = query.size()  # [4, 32, 32, 512]
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)  # [4096, 10] , [4096, 10]
        query_reshape = query.contiguous().view(batch_size * h * w, dims)  # [4096, 512]
        concat_memory = torch.matmul(softmax_score_memory.detach(), keys)  # Eq 2 -> [4096, 10], [10, 512]
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # [4096, 1024]
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)  # [4, 32, 32, 1024]
        updated_query = updated_query.permute(0, 3, 1, 2)  # [4, 1024, 32, 32]
        return updated_query, softmax_score_query, softmax_score_memory
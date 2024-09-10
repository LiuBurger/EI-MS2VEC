import torch as pt
import torch.nn as nn
import torch.nn.functional as F


class Spec2Emb(nn.Module):
    def __init__(self, num_emb:int=1000, emb_dim:int=500):
        super(Spec2Emb, self).__init__()
        self.max_exp = 6
        self.emb_con = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )
        self.emb_cen = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )
        self.trip_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def _compute_embedding(self, mzs, intens, masks, power):
        embs = self.emb_cen(mzs)
        embs = embs * masks.unsqueeze(-1)
        intens = pt.pow(intens, power).unsqueeze(-1)
        embs = (embs * intens).sum(dim=1)
        return embs

    def forward(self, data, mode:str='train', power:float=0.5):
        if mode == 'train': 
            mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg = data
            embs_con = self.emb_con(mzs_con)        # [batch, seq, emb_dim]
            embs_pos = self.emb_cen(poss_cen)     # [B, emb_dim]
            embs_neg = self.emb_cen(negs_cen)      # [B, neg_num, emb_dim]
            embs_neg *= masks_neg.unsqueeze(-1)
            # for every cen word its context words
            embs_con = embs_con[batch_idx] * masks_con.unsqueeze(-1)
            embs_con = embs_con.sum(dim=1) / masks_con.sum(dim=1).unsqueeze(-1) # [B, emb_dim]
            pos_score = (embs_con * embs_pos).sum(dim=-1) # 点积
            pos_score = pt.clamp(pos_score, max=self.max_exp, min=-self.max_exp)
            pos_score = -F.logsigmoid(pos_score)
            neg_score = pt.bmm(embs_neg, embs_con.unsqueeze(-1)).squeeze(-1) # 
            neg_score = pt.clamp(neg_score, max=self.max_exp, min=-self.max_exp)
            neg_score = -F.logsigmoid(-neg_score).sum(dim=-1)
            return (pos_score + neg_score).sum()
        
        elif mode == 'emb': # emb模式下的masks只mask掉了padding 
            mzs_all, intens_all, masks_all = data  # [batch, seq]
            return self._compute_embedding(mzs_all, intens_all, masks_all, power)
        
        elif mode == 'finetune':
            data_mea, data_pre_hit, data_pre_nhit = data
            embs_mea = self._compute_embedding(*data_mea, power)
            embs_pre_hit = self._compute_embedding(*data_pre_hit, power)
            embs_pre_nhit = self._compute_embedding(*data_pre_nhit, power)
            # batchsize, emb_dim
            embs_mea = F.normalize(embs_mea, p=2, dim=-1)
            embs_pre_hit = F.normalize(embs_pre_hit, p=2, dim=-1)
            embs_pre_nhit = F.normalize(embs_pre_nhit, p=2, dim=-1)
            # batchsize
            loss = self.trip_loss(embs_mea, embs_pre_hit, embs_pre_nhit)
            return loss
        else:
            raise ValueError('mode not exist')
        

class Linear_Scheduler:
    def __init__(self, optimizer, epochs:int, start_lr:float=0.025, end_lr:float=2.5e-4):
        self.optimizer = optimizer
        self.epochs = epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
    
    def lr_lambda(self, cur_epoch:int, epoch_progress:float):
        progress = (cur_epoch + epoch_progress) / self.epochs
        next_lr = self.start_lr - (self.start_lr - self.end_lr) * progress
        next_lr = max(next_lr, self.end_lr)
        return next_lr


# class Posi_Enco(nn.Module):
#     def __init__(self, max_len:int=1000, d_model:int=500, gpu:int=7):
#         super(Posi_Enco, self).__init__()
#         position = pt.arange(max_len).unsqueeze(-1)
#         div_term = pt.exp(pt.arange(0, d_model, 2) * -(pt.log(pt.tensor(10000.0)) / d_model))
#         pos_emb = pt.zeros((max_len, d_model), dtype=pt.float32, device=gpu)
#         pos_emb[:, 0::2] = pt.sin(position * div_term)
#         pos_emb[:, 1::2] = pt.cos(position * div_term)
#         self.pos_emb = pos_emb
    
#     def forward(self, mzs):
#         return self.pos_emb[mzs, :]
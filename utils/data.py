import numpy as np
from copy import deepcopy
import torch as pt
from torch.utils.data import Dataset
    

class SpecDataset(Dataset):
    def __init__(self, dataset, mapping=None):
        super(SpecDataset, self).__init__()
        if isinstance(dataset, list):
            self.spectra = dataset
            self.map = np.arange(len(dataset), dtype=np.int64)
        else:
            self.spectra = dataset.spectra
            self.map = mapping
    
    def __getitem__(self, idx):
        idx_ = self.map[idx]
        mzs = self.spectra[idx_].mz.astype(int).tolist()
        intens = self.spectra[idx_].intensities
        return deepcopy(mzs), deepcopy(intens)
    
    def __len__(self):
        return len(self.map)
    

class SpecDataset_finetune(Dataset):
    def __init__(self, dataset, mapping=None):
        super(SpecDataset_finetune, self).__init__()
        if isinstance(dataset, tuple):
            self.spec_mea = dataset[0]
            self.spec_pre = dataset[1]
            self.map = tuple([np.arange(len(dataset[0]), dtype=np.int64),
                             np.arange(len(dataset[1]), dtype=np.int64),
                             np.arange(len(dataset[1]), dtype=np.int64)])
        else:
            assert isinstance(mapping, tuple), 'mapping should be tuple'
            self.spec_mea = dataset.spec_mea
            self.spec_pre = dataset.spec_pre
            self.map = mapping

    def __getitem__(self, idx):
        idx_mea = self.map[0][idx]
        idx_pre_hit = self.map[1][idx]
        idx_pre_nhit = self.map[2][idx]
        mzs_mea = self.spec_mea[idx_mea].mz.astype(int).tolist()
        intens_mea = self.spec_mea[idx_mea].intensities
        mzs_pre_hit = self.spec_pre[idx_pre_hit].mz.astype(int).tolist()
        intens_pre_hit = self.spec_pre[idx_pre_hit].intensities
        mzs_pre_nhit = self.spec_pre[idx_pre_nhit].mz.astype(int).tolist()
        intens_pre_nhit = self.spec_pre[idx_pre_nhit].intensities
        return (deepcopy(mzs_mea), deepcopy(intens_mea)), \
                (deepcopy(mzs_pre_hit), deepcopy(intens_pre_hit)), \
                (deepcopy(mzs_pre_nhit), deepcopy(intens_pre_nhit))

    def __len__(self):
        return len(self.map[0])


def collate_fun(keep_prob:np.array, neg_prob:np.array, neg_num:int=5, min_len_mz:int=10, min_inten:float=0.01):
    neg_choice = np.arange(neg_prob.shape[0])
    def collate_fn(batch):
        # con: context, cen: center
        mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg = [], [], [], [], [], []
        max_len = max([len(mz) for mz, _ in batch])
        idx = 0
        for mz, inten in batch:
            len_mz = len(mz)
            if len_mz >= min_len_mz: # 移除峰的数量小于阈值的质谱 
                pad_num = max_len - len_mz
                pos_cen = []
                mask_down = np.random.random(len_mz) < keep_prob[mz]
                for i in range(len_mz):
                    if mask_down[i] and inten[i] > min_inten: # 如果没有被mask掉
                        mask_pos_down = np.array(mask_down)
                        mask_pos_down[i] = False
                        if np.any(mask_pos_down): # 上下文没有被全部mask掉
                            pos_cen.append(mz[i])
                            masks_con.append(np.pad(mask_pos_down, (0, pad_num)))
                if len(pos_cen) == 0: # 整个质谱中的中心词都被mask掉了
                    continue   
                mzs_con.append(np.pad(mz, (0, pad_num)))
                poss_cen.extend(pos_cen)
                batch_idx.extend([idx] * len(pos_cen))
                idx += 1
                neg_cen = np.random.choice(neg_choice, (len(pos_cen), neg_num), p=neg_prob)
                mask_neg = neg_cen != np.array(pos_cen)[:, np.newaxis]
                negs_cen.append(neg_cen)
                masks_neg.append(mask_neg)
        if len(mzs_con) == 0:
            return None
        mzs_con = pt.tensor(np.array(mzs_con), dtype=pt.long)
        masks_con = pt.tensor(np.array(masks_con), dtype=pt.bool)
        poss_cen = pt.tensor(np.array(poss_cen), dtype=pt.long)
        batch_idx = pt.tensor(np.array(batch_idx), dtype=pt.int)
        negs_cen = pt.tensor(np.concatenate(negs_cen), dtype=pt.long)
        masks_neg = pt.tensor(np.concatenate(masks_neg), dtype=pt.bool)
        return mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg
    return collate_fn


def collate_fun_emb(batch):
    mzs_con, intens_con, masks = [], [], []
    max_len = max([len(mz) for mz, _ in batch])
    for mz, inten in batch:
        len_mz = len(mz)
        pad_num = max_len - len_mz
        mz_con = np.pad(mz, (0, pad_num))
        inten_con = np.pad(inten, (0, pad_num))
        mask = np.pad(np.ones_like(mz, dtype=np.bool_), (0, pad_num))
        mzs_con.append(mz_con)
        intens_con.append(inten_con)
        masks.append(mask) 
    mzs_con = pt.tensor(np.array(mzs_con), dtype=pt.long)
    intens_con = pt.tensor(np.array(intens_con), dtype=pt.float)
    masks = pt.tensor(np.array(masks), dtype=pt.bool)
    return mzs_con, intens_con, masks


def collate_fun_finetune(batch):
    data_mea = [data[0] for data in batch]
    data_pre_hit = [data[1] for data in batch]
    data_pre_nhit = [data[2] for data in batch]
    return collate_fun_emb(data_mea), collate_fun_emb(data_pre_hit), collate_fun_emb(data_pre_nhit)
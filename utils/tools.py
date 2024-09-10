import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
import faiss
from datetime import datetime
import os


def gen_embeddings(model:nn.Module, loader:DataLoader, gpu:int, power:float=0.5):
    model.eval()
    embs = []
    with pt.no_grad():
        for mzs_con, intens_con, masks in loader:
            data = [d.to(gpu) for d in (mzs_con, intens_con, masks)]
            emb = model(tuple(data), mode='emb', power=power).detach().cpu().numpy()
            embs.append(emb)
    pt.cuda.empty_cache()
    embs = np.concatenate(embs, axis=0)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def build_idx(embs_lib:np.ndarray, embs_test:np.ndarray, gpu:int, topk:int=200, keep_index:bool=False):
    assert embs_lib.shape[1] == embs_test.shape[1], 'Dimension not match'
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(embs_lib.shape[1])
    index_flat = faiss.index_cpu_to_gpu(res, gpu, index_flat)
    index_flat.add(embs_lib)
    time_start = datetime.now()
    Distance, I = index_flat.search(embs_test, topk)
    print('Searching time: ', datetime.now()-time_start)
    pt.cuda.empty_cache()
    if keep_index:
        return I, Distance, index_flat
    else:
        index_flat.reset()
        return I, Distance
    

def faiss_idx(embs_lib:np.ndarray, gpu:int):
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(embs_lib.shape[1])
    index_flat = faiss.index_cpu_to_gpu(res, gpu, index_flat)
    index_flat.add(embs_lib)
    return index_flat


def calculate_hit_rate(query_mols:list, idx:list, db_mols:list, k:int=10, mode:str='inchikey'):
    # 对N个查询，有它们前10个最近邻的idx[N,10]
    N = len(query_mols)
    assert N == len(idx), 'The length of query_mols and idx should be the same.'
    top1, topk = 0, 0
    for i in range(N):
        candidate_idx = idx[i][:k]
        candidate_inchikey= [db_mols[j].metadata[mode] for j in candidate_idx]
        if query_mols[i].metadata[mode] in candidate_inchikey:
            topk += 1
            if query_mols[i].metadata[mode] == candidate_inchikey[0]:
                top1 += 1
    return top1/N, topk/N


def evaluate(mols_test:list, I:list, mols_all:list, f=None, library_type:str='insilico'):
    top1, top10 = calculate_hit_rate(mols_test, I, mols_all)
    print(f'{library_type} library')
    print(f'Top1 hit rate: {100*top1:.2f}%\nTop10 hit rate: {100*top10:.2f}%')
    if f is not None:
        f.write(f'{library_type} library\n')
        f.write(f'Top1 hit rate: {100*top1:.2f}%\nTop10 hit rate: {100*top10:.2f}%\n')
    return top1, top10


def find_nearest_hit_nhit(I:list, mols_val:list, mols_all:list):
    assert len(I) == len(mols_val), 'Length not match'
    vals, hits, nhits = [], [], []
    for i in range(len(I)):
        candidate_idx = I[i]
        inchikey = mols_val[i].metadata['inchikey']
        hit = None
        for j in candidate_idx:
            if inchikey == mols_all[j].metadata['inchikey']:
                hit = j
                break
        if hit is not None:
            for j in candidate_idx:
                if inchikey != mols_all[j].metadata['inchikey']:
                    nhits.append(j)
                    break
            vals.append(i)
            hits.append(hit)
    return np.array(vals), np.array(hits), np.array(nhits)


def hit_rate_topk(query_mols:list, idx:list, db_mols:list, topk:int=100, step:int=10, mode:str='inchikey'):
    N = len(query_mols)
    assert N == len(idx), 'The length of query_mols and idx should be the same.'
    assert topk % step == 0, 'topk should be divisible by step.'
    topk_hit = pt.zeros(topk//step)
    for i in range(N):
        candidate_inchikey = []
        for k in range(0, topk, step):
            candidate_idx_new = idx[i][k:k+step]
            candidate_inchikey_new = [db_mols[j].metadata[mode] for j in candidate_idx_new]
            candidate_inchikey.extend(candidate_inchikey_new)
            candidate_inchikey_set = set(candidate_inchikey)
            if query_mols[i].metadata[mode] in candidate_inchikey_set:
                topk_hit[(k+step)//step-1] += 1 
    return topk_hit/N


def mass_filter_5Da(query_mols:list, idx:list, db_mols:list, mode:str='inchikey'):
    N = len(query_mols)
    assert N == len(idx), 'The length of query_mols and idx should be the same.'
    top1, topk = 0, 0
    for i in range(N):
        candidate_idx = idx[i]
        candidate_idx_5Da = []
        for j in candidate_idx:
            if abs(float(query_mols[i].metadata['nominal_mass']) - float(db_mols[j].metadata['nominal_mass'])) <= 5:
                candidate_idx_5Da.append(j)
                if len(candidate_idx_5Da) == 10: # 保证了一定是Top10
                    break
        candidate_inchikey= [db_mols[j].metadata[mode] for j in candidate_idx_5Da]
        if query_mols[i].metadata[mode] in candidate_inchikey:
            topk += 1
            if query_mols[i].metadata[mode] == candidate_inchikey[0]:
                top1 += 1
    return top1/N, topk/N


def save_model(model:nn.Module, model_name:str, epoch:int):
    if not os.path.exists(f'./model/{model_name}_epoch{epoch+1}.pth'):
        pt.save(model.state_dict(), f'./model/{model_name}_epoch{epoch+1}.pth')
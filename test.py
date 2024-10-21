import torch as pt
import numpy as np
from torch.utils.data import DataLoader
from utils.data import SpecDataset, collate_fun_emb
from utils.model import Spec2Emb
from utils.tools import gen_embeddings, build_idx, evaluate
from argparse import ArgumentParser


def test(args):
    print('Loading data...')
    mols_test = pt.load('./data/mine/test_11499.pt')
    mols_all = pt.load('./data/mine/mols_all.pt')
    mass_all = np.array([float(mol.metadata['nominal_mass']) for mol in mols_all])
    mass_test = np.array([float(mol.metadata['nominal_mass']) for mol in mols_test])
    dataset_lib = SpecDataset(mols_all)
    loader_lib = DataLoader(dataset_lib, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    dataset_test = SpecDataset(mols_test)
    loader_test = DataLoader(dataset_test, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    
    print('Testing...')
    gpu = args.gpu
    model = Spec2Emb().to(gpu)
    model.load_state_dict(pt.load(args.model_path, map_location='cpu'))
    embeddings_lib = gen_embeddings(model, loader_lib, gpu, power=0.4) 
    embeddings_test = gen_embeddings(model, loader_test, gpu, power=0.4)
    I_expand, _ = build_idx(embeddings_lib, embeddings_test, gpu, topk=200)
    top1_expand, top10_expand = evaluate(mols_test, I_expand, mols_all, library_type='expanded')
    I_insilico, _ = build_idx(embeddings_lib[:2146690], embeddings_test, gpu, topk=200)
    top1_insilico, top10_insilico = evaluate(mols_test, I_insilico, mols_all, library_type='insilico')
    print(f'\nWith Mass:')
    embeddings_lib[:, -1] = mass_all
    embeddings_test[:, -1] = mass_test
    I_expand, _ = build_idx(embeddings_lib, embeddings_test, gpu, topk=200)
    top1_expand, top10_expand = evaluate(mols_test, I_expand, mols_all, library_type='expanded')
    I_insilico, _ = build_idx(embeddings_lib[:2146690], embeddings_test, gpu, topk=200)
    top1_insilico, top10_insilico = evaluate(mols_test, I_insilico, mols_all, library_type='insilico')

def main():
    parser = ArgumentParser(description='Finetune model')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='./model/mass_ft_p0.4_epoch2.pth')
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()

# nohup python -Bu test.py > test.out
import torch as pt
import numpy as np
from torch.utils.data import DataLoader
from utils.data import SpecDataset, SpecDataset_finetune, collate_fun_emb, collate_fun_finetune
import torch.optim as optim
from utils.model import Spec2Emb
from tqdm import tqdm
from utils.tools import gen_embeddings, build_idx, evaluate, find_nearest_hit_nhit, save_model
from argparse import ArgumentParser


def finetune(args):
    print('Loading data...')
    mols_test = pt.load('./data/mine/test_11499.pt')
    mols_val = pt.load('./data/mine/val_11825.pt')
    mols_all = pt.load('./data/mine/mols_all.pt')
    mass_all = np.array([float(mol.metadata['nominal_mass']) for mol in mols_all])
    mass_test = np.array([float(mol.metadata['nominal_mass']) for mol in mols_test])
    mass_val = np.array([float(mol.metadata['nominal_mass']) for mol in mols_val])
    dataset_lib = SpecDataset(mols_all)
    loader_lib = DataLoader(dataset_lib, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    dataset_val = SpecDataset(mols_val)
    loader_val = DataLoader(dataset_val, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    dataset_test = SpecDataset(mols_test)
    loader_test = DataLoader(dataset_test, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    dataset_finetune = SpecDataset_finetune((mols_val, mols_all))

    print('Building model')
    gpu = args.gpu
    model = Spec2Emb().to(gpu)
    model.load_state_dict(pt.load(args.pretrain_model, map_location='cpu'))
    epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    f = open(args.file_name+'.txt', 'w')
    model_name = args.model_name
    max_metrics = {'expanded': [0, 0], 'insilico': [0, 0], 'expanded_mass': [0, 0], 'insilico_mass': [0, 0]}

    print('Start finetuning...')
    for epoch in range(epochs):  
        print(f'==================================Finetune_epoch{epoch+1}======================================')
        f.write('\nFinetune_epoch%d\n' % (epoch+1))
        embeddings_lib = gen_embeddings(model, loader_lib, gpu, power=0.4)
        embeddings_val = gen_embeddings(model, loader_val, gpu, power=0.4)
        embeddings_lib[:, -1] = mass_all
        embeddings_val[:, -1] = mass_val
        I, _ = build_idx(embeddings_lib, embeddings_val, gpu, topk=200) # 内置清缓存
        top1_val, top10_val = evaluate(mols_val, I, mols_all, f, 'Validation')
        vals, hits, nhits = find_nearest_hit_nhit(I, mols_val, mols_all)
        dataset_ft = SpecDataset_finetune(dataset_finetune, mapping=(vals, hits, nhits))
        loader_ft = DataLoader(dataset_ft, args.ft_batch, shuffle=True, 
                               num_workers=args.num_workers, collate_fn=collate_fun_finetune)
        model.train()
        for j in range(5):
            finetune_loss = []
            for i, Data in enumerate(tqdm(loader_ft, unit='batch')):
                Data = [d.to(gpu) for data in Data for d in data]
                optimizer.zero_grad()
                loss = model((Data[:3], Data[3:6], Data[6:9]), mode='finetune', power=0.4)
                finetune_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                if (i+1) %300 ==0:
                    loss = np.mean(finetune_loss)
                    print(f'Total Loss: {loss}')
                    finetune_loss = []

        print(f'===================================Test_epoch{epoch+1}======================================')
        f.write('\n\nTest_epoch%d\n' % (epoch+1))
        embeddings_lib = gen_embeddings(model, loader_lib, gpu, power=0.4) 
        embeddings_test = gen_embeddings(model, loader_test, gpu, power=0.4)
        I_expand, _ = build_idx(embeddings_lib, embeddings_test, gpu, topk=200)
        top1_expand, top10_expand = evaluate(mols_test, I_expand, mols_all, f, 'expanded')
        if top1_expand > max_metrics['expanded'][0] and top10_expand > max_metrics['expanded'][1]:
            max_metrics['expanded'] = [top1_expand, top10_expand]
            save_model(model, model_name, epoch)
        I_insilico, _ = build_idx(embeddings_lib[:2146690], embeddings_test, gpu, topk=200)
        top1_insilico, top10_insilico = evaluate(mols_test, I_insilico, mols_all, f, 'insilico')
        if top1_insilico > max_metrics['insilico'][0] and top10_insilico > max_metrics['insilico'][1]:
            max_metrics['insilico'] = [top1_insilico, top10_insilico]
            save_model(model, model_name, epoch)
        print(f'\nWith Mass:')
        f.write('With Mass:\n')
        embeddings_lib[:, -1] = mass_all
        embeddings_test[:, -1] = mass_test
        I_expand, _ = build_idx(embeddings_lib, embeddings_test, gpu, topk=200)
        top1_expand, top10_expand = evaluate(mols_test, I_expand, mols_all, f, 'expanded')
        if top1_expand > max_metrics['expanded_mass'][0] and top10_expand > max_metrics['expanded_mass'][1]:
            max_metrics['expanded_mass'] = [top1_expand, top10_expand]
            save_model(model, model_name, epoch)
        I_insilico, _ = build_idx(embeddings_lib[:2146690], embeddings_test, gpu, topk=200)
        top1_insilico, top10_insilico = evaluate(mols_test, I_insilico, mols_all, f, 'insilico')
        if top1_insilico > max_metrics['insilico_mass'][0] and top10_insilico > max_metrics['insilico_mass'][1]:
            max_metrics['insilico_mass'] = [top1_insilico, top10_insilico]
            save_model(model, model_name, epoch)
        print(f'================================================================================================')
    f.close()


def main():
    parser = ArgumentParser(description='Finetune model')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--ft_batch', type=int, default=32)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--file_name', type=str, default='finetune')
    parser.add_argument('--pretrain_model', type=str, default='pretrain.pth')
    parser.add_argument('--model_name', type=str, default='finetune')
    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()

# nohup python -Bu finetune.py > finetune.out
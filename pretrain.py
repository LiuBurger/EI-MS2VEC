import torch as pt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.model import Spec2Emb, Linear_Scheduler
from utils.data import SpecDataset, collate_fun, collate_fun_emb
from tqdm import tqdm
from utils.tools import gen_embeddings, build_idx, evaluate, save_model
from argparse import ArgumentParser


def train(args):
    print('Loading data...')
    mols_test = pt.load('./data/mine/test_11499.pt')
    mols_all = pt.load('./data/mine/mols_all.pt')
    # 统计词频
    mols_train = mols_all[:232826]
    count_list = np.zeros(1000)
    for mol in mols_train:
        tmp_list = np.zeros(1000)
        for mz in mol.mz:
            tmp_list[int(mz)] = 1
        count_list += tmp_list
    count_list += 1  
    # 生成负采样概率
    pow_frequency = np.array(count_list) ** 0.75
    neg_prob = pow_frequency / pow_frequency.sum()
    # 生成下采样概率
    mzs_freq = np.array(count_list)
    mzs_freq = mzs_freq / np.sum(mzs_freq)
    t = 1e-3
    keep_prob = np.array([np.sqrt(t/f) + t/f for f in mzs_freq])

    dataset_lib = SpecDataset(mols_all)
    dataset_test = SpecDataset(mols_test)
    loader_lib = DataLoader(dataset_lib, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    loader_test = DataLoader(dataset_test, batch_size=args.test_batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fun_emb)
    dataset_train = SpecDataset(dataset_lib, mapping=np.arange(232826))
    loader_train = DataLoader(dataset_train, batch_size=args.train_batch, shuffle=True, 
                                num_workers=args.num_workers, collate_fn=collate_fun(keep_prob, neg_prob))
    num_batches = len(loader_train)

    print('Building model')
    gpu = args.gpu
    model = Spec2Emb().to(gpu)
    epochs = args.epochs
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = Linear_Scheduler(optimizer, epochs, start_lr=lr, end_lr=args.end_lr)
    f = open(args.file_name+'.txt', 'w')
    model_name = args.model_name
    max_metrics = {'expand': [0, 0], 'insilico': [0, 0]}

    print('Start training...')
    for epoch in range(epochs):
        print(f'==================================Train_epoch{epoch+1}======================================')
        model.train()
        train_loss = []
        for i, Data in enumerate(tqdm(loader_train, unit='batch')):
            data = [d.to(gpu) for d in Data]
            optimizer.zero_grad()
            loss = model(data)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            batch_progress = (i+1)/num_batches
            lr = scheduler.lr_lambda(epoch, batch_progress)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if (i+1) %1000 ==0:
                loss = np.mean(train_loss)
                print(f'Total Loss: {loss}')
                train_loss = []
        
        print(f'===================================Test_epoch{epoch+1}======================================')
        f.write('\nTest_epoch%d\n' % (epoch+1))
        embeddings_lib = gen_embeddings(model, loader_lib, gpu)
        embeddings_test = gen_embeddings(model, loader_test, gpu)
        I_expand, _ = build_idx(embeddings_lib, embeddings_test, gpu)
        top1_expand, top10_expand = evaluate(mols_test, I_expand, mols_all, f, 'Expanded')
        if top1_expand > max_metrics['expand'][0] and top10_expand > max_metrics['expand'][1]:
            max_metrics['expand'] = [top1_expand, top10_expand]
            save_model(model, model_name, epoch)
        I_insilico, _ = build_idx(embeddings_lib[:2146690], embeddings_test, gpu)
        top1_insilico, top10_insilico = evaluate(mols_test, I_insilico, mols_all, f, 'Insilico')
        if top1_insilico > max_metrics['insilico'][0] and top10_insilico > max_metrics['insilico'][1]:
            max_metrics['insilico'] = [top1_insilico, top10_insilico]
            save_model(model, model_name, epoch)
        print(f'================================================================================================')
    f.close()
    print('Done!')


def main():
    parser = ArgumentParser(description='Pretrain model')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--train_batch', type=int, default=32)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--end_lr', type=float, default=2.5e-4)
    parser.add_argument('--file_name', type=str, default='pretrain')
    parser.add_argument('--model_name', type=str, default='pretrain')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

# nohup python -Bu pretrain.py > pretrain.out
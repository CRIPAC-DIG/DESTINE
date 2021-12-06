# import apex
import pretty_errors
import tqdm
import torch
import argparse
from time import perf_counter
import hashlib

from utils import EarlyStopper, hash_dict
from dataloader import get_data
from models.DESTINE import DESTINE
from simple_param.sp import SimpleParam

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchfm.model.afi import AutomaticFeatureInteractionModel
from layers.activation import \
    SelfAttention, DisentangledSelfAttention, DisentangledSelfAttentionAverage, \
    PairwiseSelfAttention, UnarySelfAttention, \
    DisentangledSelfAttentionVariant1, DisentangledSelfAttentionVariant2, \
    DisentangledSelfAttentionWeighted, DisentangledSelfAttentionAverageLearnable, \
    ScaledDisentangledSelfAttention


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    losses = []
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

    return sum(losses) / len(losses)


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    num_samples = 0
    total_loss = 0.
    log_loss = torch.nn.BCELoss(reduction='sum')
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            loss = log_loss(y, target.float())
            num_samples += target.size(0)
            total_loss += loss.item()
    return total_loss / num_samples, roc_auc_score(targets, predicts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12321)
    parser.add_argument('--dataset', default='avazu')
    parser.add_argument('--param', type=str, default='default')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--load_dataset', type=str, nargs='?')
    parser.add_argument('--save_dataset', type=str, nargs='?')
    parser.add_argument('--save_dir', default='checkpoints')
    default_param = {
        'epoch': 100,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'batch_size': 10000,
        'embed_dim': 40,
        'attn_embed_dim': 80,
        'mlp_dim': 400,
        'weight_decay': 5e-6,
        'res_mode': 'last_layer',
        'scale_att': True,
        'relu_before_att': False,
        'num_heads': 2,
        'num_layers': 3,
        'base_model': 'DSelfAttn',
        'deep': False,
        'magic': False
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            if type(default_param[key]) is bool:
                param[key] = getattr(args, key) == 'True'
            else:
                param[key] = type(default_param[key])(getattr(args, key))

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    print(args)
    print(param)
    param_hash = hash_dict(
        {**param, 'seed': args.seed, 'split': args.load_dataset if args.load_dataset is not None else 'random'}
    )
    print(f'exp hash code {param_hash}')

    models = {
        'SelfAttn': SelfAttention,
        'DSelfAttn': DisentangledSelfAttention,
        'DSelfAttnAvg': DisentangledSelfAttentionAverage,
        'Pairwise': PairwiseSelfAttention,
        'Unary': UnarySelfAttention,
        'DSelfAttn1': DisentangledSelfAttentionVariant1,
        'DSelfAttn2': DisentangledSelfAttentionVariant2,
        'DSelfAttnWgt': DisentangledSelfAttentionWeighted,
        'DSelfAttnWgtL': DisentangledSelfAttentionAverageLearnable,
        'ScaledDSelfAttn': ScaledDisentangledSelfAttention
    }

    base_model = models[param['base_model']]

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    dataset = get_data(args.dataset)
    device = torch.device(args.device)
    model = DESTINE(
        dataset.field_dims, embed_dim=param['embed_dim'], atten_embed_dim=param['num_heads'] * param['embed_dim'],
        num_heads=param['num_heads'], num_layers=param['num_layers'], mlp_dims=(param['mlp_dim'], param['mlp_dim']),
        dropout_mlp=param['dropout'], dropout_att=param['dropout'], base_model=base_model,
        scale_att=param['scale_att'],
        relu_before_att=param['relu_before_att'],
        res_mode=param['res_mode'],
        deep=param['deep'],
        magic=param['magic']
    ).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    if args.load_dataset:
        (train_dataset_indices, valid_dataset_indices, test_dataset_indices) = torch.load(args.load_dataset)

        train_dataset = torch.utils.data.Subset(dataset, train_dataset_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_dataset_indices)
        valid_dataset = torch.utils.data.Subset(dataset, valid_dataset_indices)

    elif args.save_dataset:
        torch.save((train_dataset.indices, valid_dataset.indices, test_dataset.indices), args.save_dataset)
        exit(0)

    train_data_loader = DataLoader(train_dataset, batch_size=param['batch_size'], num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=param['batch_size'], num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=param['batch_size'], num_workers=8)

    early_stopper = EarlyStopper(num_trials=2, save_path=f'{args.save_dir}/{args.dataset}_{param_hash[:8]}.pt')
    epoch_time = []
    for epoch_i in range(param['epoch']):
        tic = perf_counter()
        train(model, optimizer, train_data_loader, criterion, device)
        logloss, auc = test(model, valid_data_loader, device)
        scheduler.step(logloss)
        toc = perf_counter()
        print('epoch:', epoch_i, 'validation: auc:', auc, f'logloss: {logloss}', f'time: {toc - tic:.6f} sec')
        epoch_time.append(toc - tic)

        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break

    avg = lambda xs: sum(xs) / len(xs)
    best_model = early_stopper.load_best().to(device)
    # auc = test(model, test_data_loader, device)
    logloss, auc = test(best_model, test_data_loader, device)
    print(f'test auc {auc}, test logloss {logloss}')
    print(f'training time {param["epoch"]}x{avg(epoch_time)} sec')

    # model.eval()
    # with torch.no_grad():
    #     temp_loader = DataLoader(test_dataset, batch_size=test_length)
    #     for fields, _ in temp_loader:
    #         fields = fields.to(device)
    #         attention_map = model.get_attention_map(fields)
    #         torch.save(
    #             attention_map,
    #             f'data/{args.dataset}_{args.seed}_attn.pt')

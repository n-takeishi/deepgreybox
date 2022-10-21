import os
import json
import sys
import yaml
import numpy as np
from tqdm import tqdm
import torch

from train import loss_fun


def main(args):
    with open(os.path.join(args.exptname, 'params.yaml'), 'r') as fd:
        params_model = yaml.safe_load(fd)['model']

    device = torch.device(args.device)

    print('exptname:', args.exptname)
    print('mode:', args.mode)
    print('seed:', args.seed)
    print('device:', device)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load training dataset (may be reused here)
    datadir = os.path.join(args.exptname, 'out', args.trialname)
    with np.load(os.path.join(datadir, 'data_tr.npz')) as data:
        data_tr_only_x = torch.Tensor(data['x'])
        dataset_tr_only_x = torch.utils.data.TensorDataset(data_tr_only_x)
        dataset_tr = torch.utils.data.TensorDataset(
            torch.Tensor(data['x']), torch.Tensor(data['y'])
        )

    # load prediction dataset
    with np.load(os.path.join(datadir, 'data_%s.npz'%args.target)) as data:
        if 'thT' in data:
            thT_truth = data['thT']
        else:
            thT_truth = np.empty((data['x'].shape[0], 0))
        data_pred_only_x = torch.Tensor(data['x'])
        dataset_pred_only_x = torch.utils.data.TensorDataset(data_pred_only_x)
        dataset_pred = torch.utils.data.TensorDataset(
            torch.Tensor(data['x']), torch.Tensor(data['y']), torch.Tensor(thT_truth)
        )

    # result directory
    outdir = os.path.join(args.exptname, 'out', args.trialname, args.mode)

    # load model
    sys.path.insert(0, args.exptname)
    from model import GreyboxModel
    model = GreyboxModel(params_model).to(device)
    model.load_state_dict(
        torch.load(os.path.join(outdir, 'model_tr.pt'),
        map_location=device)
    )

    # prepare to-be-saved dict
    curves = {'L': [], 'R':[]}

    # main prediction-time optimization process of each mode
    model.train()
    if args.mode == 'adaptive' and 'prm_thT_fixed' in {name:p for name, p in model.named_parameters()} and len(model.thT_bound)==1:
        model.eval()

        # set data
        _data = data_pred_only_x if args.adaptive_pred_only else torch.cat([data_tr_only_x, data_pred_only_x], dim=0)

        # grid search
        grid = np.linspace(model.thT_bound[0][0], model.thT_bound[0][1], args.num_grids)
        R = []
        with torch.no_grad():
            with tqdm(range(args.num_grids), desc='predict grid', leave=False) as pbar:
                for i in pbar:
                    _thT = torch.tensor([grid[i]]).float().view(1,1).expand(_data.shape[0],1)
                    _, _R = model(_data, thT=_thT)
                    R.append(_R.item())

                    pbar.set_postfix({
                        'R': '%0.2e'%_R.item()
                    })
        
        minidx = np.argmin(R)
        model.prm_thT_fixed.data = torch.tensor([grid[minidx]]).float().view(1,1)

        print('grid search done')
        print('minimum R:', R[minidx])

    elif args.mode == 'adaptive':
        # set dataloader
        concat_dataset = torch.utils.data.ConcatDataset([dataset_tr_only_x, dataset_pred_only_x])
        loader_tr_and_pred_only_x = torch.utils.data.DataLoader(
            dataset_pred_only_x if args.adaptive_pred_only else concat_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        # set optimizer and learning rate scheduler; only thT-related things will be updated
        parameters = []
        for name, p in model.named_parameters():
            if name == 'prm_thT_fixed' or name.startswith('net_thT_encoder'):
                parameters.append(p)

        optimizer = torch.optim.AdamW(
            parameters,
            lr = args.learning_rate[0],
            weight_decay = args.weight_decay,
            betas = (
                args.adam_beta1 if hasattr(args, 'adam_beta1') else 0.9,
                args.adam_beta2 if hasattr(args, 'adam_beta2') else 0.999
            )
        )

        gamma = np.exp(
            (np.log(args.learning_rate[1])-np.log(args.learning_rate[0])) \
                / args.max_epochs
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # main iteration
        for epoch in range(args.max_epochs):
            R_list = []
            with tqdm(loader_tr_and_pred_only_x, desc='%s epoch %d'%('predict', epoch), leave=False) as pbar:
                for batch in pbar:
                    x = batch[0].to(device)
                    model.zero_grad()
                    _, R = model(x, num_thT_samples_per_x=0)
                    R.backward()
                    torch.nn.utils.clip_grad_value_(
                        optimizer.param_groups[0]['params'],
                        args.grad_clip_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        optimizer.param_groups[0]['params'],
                        args.grad_clip_norm
                    )
                    optimizer.step()

                    # projection to feasible range of thT_fixed
                    if hasattr(model, 'prm_thT_fixed'):
                        with torch.no_grad():
                            for j,bound in enumerate(model.thT_bound):
                                model.prm_thT_fixed[:,j].clamp_(bound[0], bound[1])

                    # log
                    pbar.set_postfix({
                        'R': '%0.2e'%R.item()
                    })
                    R_list.append(R.item())

            curves['R'].append(R_list)

            if scheduler is not None:
                scheduler.step()

        print('prediction-time optimization done:', epoch+1, 'epochs')
        print('final R:', np.mean(curves['R'][-1]).item())

    elif args.mode == 'maml':
        # set dataloader
        concat_dataset = torch.utils.data.ConcatDataset([dataset_tr_only_x, dataset_pred_only_x])
        loader_tr_and_pred_only_x = torch.utils.data.DataLoader(
            concat_dataset,
            batch_size=len(concat_dataset),
            shuffle=True
        ) # NOTE: only inner_num_iters iterations are to be run

        # set optimizer and learning rate scheduler; all parameters will be updated
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.maml_inner_learning_rate
        )

        # main iteration
        for epoch in range(args.maml_inner_num_iters): # only inner_num_iters iterations are to be run
            R_list = []
            with tqdm(loader_tr_and_pred_only_x, desc='%s epoch %d'%('predict', epoch), leave=False) as pbar:
                for batch in pbar:
                    x = batch[0].to(device)
                    model.zero_grad()
                    _, R = model(x)
                    R.backward()
                    torch.nn.utils.clip_grad_value_(
                        optimizer.param_groups[0]['params'],
                        args.maml_inner_grad_clip_value
                    )
                    optimizer.step()

                    # projection to feasible range of thT_fixed
                    if hasattr(model, 'prm_thT_fixed'):
                        with torch.no_grad():
                            for j,bound in enumerate(model.thT_bound):
                                model.prm_thT_fixed[:,j].clamp_(bound[0], bound[1])

                    # log
                    pbar.set_postfix({
                        'R': '%0.2e'%R.item()
                    })
                    R_list.append(R.item())

            curves['R'].append(R_list)

        print('prediction-time optimization done:', epoch+1, 'epochs')
        print('final R:', np.mean(curves['R'][-1]).item())

    elif args.mode == 'transductive':
        # set dataloaders
        loader_tr = torch.utils.data.DataLoader(
            dataset_tr,
            batch_size=args.batch_size,
            shuffle=True
        )
        loader_pred_only_x = torch.utils.data.DataLoader(
            dataset_pred_only_x,
            batch_size=args.batch_size,
            shuffle=True
        )

        # set optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.learning_rate[0],
            weight_decay = args.weight_decay,
            betas = (
                args.adam_beta1 if hasattr(args, 'adam_beta1') else 0.9,
                args.adam_beta2 if hasattr(args, 'adam_beta2') else 0.999
            )
        )
        gamma = np.exp(
            (np.log(args.learning_rate[1])-np.log(args.learning_rate[0])) \
                / args.max_epochs
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # main iteration
        for epoch in range(args.max_epochs):
            L_list = []
            R_list = []
            loader_pred_only_x_iterator = iter(loader_pred_only_x)
            with tqdm(loader_tr, desc='%s epoch %d'%('predict', epoch), leave=False) as pbar:
                for batch_tr in pbar:
                    # load from one of the dataloaders
                    x_tr = batch_tr[0].to(device)
                    y_tr = batch_tr[1].to(device)
                    # load from another dataloader
                    try:
                        batch_pred_only_x = next(loader_pred_only_x_iterator)
                    except StopIteration:
                        loader_pred_only_x_iterator = iter(loader_pred_only_x)
                        batch_pred_only_x = next(loader_pred_only_x_iterator)
                    x_pred = batch_pred_only_x[0].to(device)
                    # combine x
                    x_tr_and_pred = torch.cat([x_tr, x_pred], dim=0)

                    model.zero_grad()
                    yh_tr_and_pred, R = model(x_tr_and_pred)
                    yh_tr = yh_tr_and_pred[:x_tr.shape[0]]
                    L = loss_fun(yh_tr, y_tr)
                    (L + float(args.coeff_R)*R).backward()
                    torch.nn.utils.clip_grad_value_(
                        optimizer.param_groups[0]['params'],
                        args.grad_clip_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        optimizer.param_groups[0]['params'],
                        args.grad_clip_norm
                    )
                    optimizer.step()

                    # projection to feasible range of thT_fixed
                    if hasattr(model, 'prm_thT_fixed'):
                        with torch.no_grad():
                            for j,bound in enumerate(model.thT_bound):
                                model.prm_thT_fixed[:,j].clamp_(bound[0], bound[1])

                    # log
                    pbar.set_postfix({
                        'L': '%0.2e'%L.item(),
                        'R': '%0.2e'%R.item()
                    })
                    L_list.append(L.item())
                    R_list.append(R.item())

            curves['L'].append(L_list)
            curves['R'].append(R_list)

            # update learning rate
            scheduler.step()

        print('prediction-time optimization done:', epoch+1, 'epochs')
        print('final L:', np.mean(curves['L'][-1]).item())
        print('final R:', np.mean(curves['R'][-1]).item())

    elif args.mode == 'inductive':
        print('inductive mode: nothing is done')

    else:
        raise ValueError('unknown mode')

    if hasattr(model, 'prm_thT_fixed'):
        print('final thT_fixed:', model.prm_thT_fixed.cpu().detach().numpy())

    # save model at last step
    torch.save(model.state_dict(), os.path.join(outdir, 'model_%s.pt'%args.target))

    # save optimization curves
    for k, v in curves.items():
        curves[k] = np.array(v)
    np.savez(os.path.join(outdir, 'curve_%s.npz'%args.target), **curves)

    # -------------------------------------------------------------

    # set dataloader for prediction
    loader_pred = torch.utils.data.DataLoader(
        dataset_pred,
        batch_size=args.pred_batch_size if hasattr(args, 'pred_batch_size') else len(dataset_pred),
        shuffle=False
    )

    # prediction
    model.eval()
    L_list = []
    R_list = []
    thTerr_list = []
    with torch.no_grad():
        with tqdm(loader_pred, leave=False) as pbar:
            for batch in pbar:
                x = batch[0].to(device)
                y = batch[1].to(device)
                thT_truth = batch[2].to(device)

                yh, R = model(x, num_thT_samples_per_x=0)

                L = loss_fun(yh, y)
                L_list.append(L.item())
                R_list.append(R.item())

                if thT_truth.shape[1] > 0:
                    if hasattr(model, 'prm_thT_fixed'):
                        thT = model.prm_thT_fixed
                    elif hasattr(model, 'net_thT_encoder') and hasattr(model, 'thT_encode'):
                        thT = model.thT_encode(x)
                    else:
                        raise ValueError('no way to get thT is defined')
                    thTerr = torch.nn.functional.mse_loss(
                        thT.expand(thT_truth.shape[0],-1),
                        thT_truth,
                        reduction='mean'
                    )
                    thTerr_list.append(thTerr.item())

    print('prediction done')
    print('prediction L:', np.mean(L_list).item())
    print('prediction R:', np.mean(R_list).item())
    print('prediction thTerr:', np.mean(thTerr_list).item())

    # save metric
    metrics = {
        'L_pred': np.mean(L_list).item(),
        'R_pred': np.mean(R_list).item(),
        'thTerr_pred': np.mean(thTerr_list).item()
    }
    with open(os.path.join(outdir, 'metric_%s.json'%args.target), 'w') as f:
        json.dump(metrics, f)

    # save args
    with open(os.path.join(outdir, 'args_%s.json'%args.target), 'w') as f:
        json.dump(vars(args), f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('exptname', type=str)
    parser.add_argument('trialname', type=str)
    parser.add_argument('--mode', type=str, choices=['inductive', 'transductive', 'adaptive'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--target', type=str, choices=['va','te'], required=True)
    parser.add_argument('--coeff-R', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=999999)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, nargs=2, default=[1e-2, 1e-5])
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--grad-clip-value', type=float, default=1.0)
    parser.add_argument('--grad-clip-norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--maml-inner-num-iters', type=int, default=5)
    parser.add_argument('--maml-inner-learning-rate', type=float, default=1e-2)
    parser.add_argument('--maml-grad-clip-value', type=float, default=999999)
    parser.add_argument('--num-grids', type=int, default=2000)
    parser.add_argument('--adaptive-pred-only', action='store_true')

    args = parser.parse_args()

    if args.mode=='transductive' and args.coeff_R==None:
        raise ValueError('For mode=transductive, coeff_R must be specified.')

    main(args)

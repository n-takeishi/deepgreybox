import sys
import os
import pathlib
import json
import yaml
import numpy as np
from tqdm import tqdm
import torch

# from clone_module import clone_module


def loss_fun(pred, target):
    return torch.nn.functional.mse_loss(pred, target, reduction='mean')


def run_epoch(epochmode, epoch, loader, model, optimizer, args):
    device = torch.device(args.device)

    if epochmode == 'train':
        model.train()
    elif epochmode == 'valid':
        model.eval()
    else:
        raise ValueError('unknown epoch mode: %s'%epochmode)

    L_list = []
    R_list = []
    with tqdm(loader, desc='%s epoch %d'%(epochmode, epoch), leave=False) as pbar:
        for batch in pbar:
            x = batch[0].to(device)
            y = batch[1].to(device)

            model.zero_grad()

            # get objectives
            def _step():
                if args.mode == 'adaptive':
                    k = args.num_thT_samples_per_x
                    yh, R = model(x, num_thT_samples_per_x=k)
                    if k > 0:
                        _y = y.repeat(k, *[1 for j in range(y.ndim-1)])
                    else:
                        _y = y
                    L = loss_fun(yh, _y)

                elif args.mode == 'inductive':
                    yh, R = model(x)
                    L = loss_fun(yh, y)

                elif args.mode == 'maml':
                    # clone the model for inner loop
                    model_cloned = clone_module(model)

                    # MAML's inner loop to update the cloned model
                    for inner_iter in range(args.maml_inner_num_iters):
                        _, R = model_cloned(x)
                        grads = torch.autograd.grad(
                            [R],
                            list(model_cloned.parameters()),
                            create_graph=False, # 1st-order MAML
                            allow_unused=True
                        )
                        with torch.no_grad():
                            for i, p in enumerate(model.parameters()):
                                grad = grads[i]
                                if grads is None:
                                    continue
                                grad.clamp_(
                                    -1.0*args.maml_inner_grad_clip_value,
                                    args.maml_inner_grad_clip_value
                                )
                                new_p = p - args.maml_inner_learning_rate*grad
                                p.copy_(new_p)
                                # 2nd-order MAML does not work with this .copy_()

                    # compute loss with the cloned model, and then do backprop,
                    # which goes back also to the original model because of the cloning
                    yh, R = model_cloned(x)
                    L = loss_fun(yh, y)

                return L, R

            if epochmode == 'train' or args.mode == 'maml':
                L, R = _step()
            elif epochmode == 'valid' and args.mode != 'maml':
                with torch.no_grad():
                    L, R = _step()

            # update model parameters
            if epochmode == 'train':
                (L + float(args.coeff_R)*R).backward()
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    args.grad_clip_value
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
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

    return L_list, R_list


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

    # load datasets
    datadir = os.path.join(args.exptname, 'out', args.trialname)
    with np.load(os.path.join(datadir, 'data_tr.npz')) as data:
        dataset_tr = torch.utils.data.TensorDataset(torch.Tensor(data['x']), torch.Tensor(data['y']))
    with np.load(os.path.join(datadir, 'data_va.npz')) as data:
        dataset_va = torch.utils.data.TensorDataset(torch.Tensor(data['x']), torch.Tensor(data['y']))

    # set model
    sys.path.insert(0, args.exptname)
    from model import GreyboxModel
    model = GreyboxModel(params_model).to(device)

    # prepare directory to save results
    outdir = os.path.join(args.exptname, 'out', args.trialname, args.mode)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # prepare to-be-saved dicts
    curves = {'L_tr':[], 'L_va':[], 'R_tr':[], 'R_va':[]}
    metrics = {'L_tr':float('inf'), 'L_va':float('inf')}

    # main training process of each mode
    if args.mode == 'adaptive' or args.mode == 'inductive' or args.mode == 'maml':
        # set dataloaders
        loader_tr = torch.utils.data.DataLoader(
            dataset_tr, batch_size=args.batch_size, shuffle=True
        )
        loader_va = torch.utils.data.DataLoader(
            dataset_va, batch_size=args.batch_size, shuffle=False
        )

        # set optimizer(s) and learning rate scheduler
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
            # train
            L_list_tr, R_list_tr = run_epoch(
                'train', epoch, loader_tr, model, optimizer, args
            )
            curves['L_tr'].append(L_list_tr)
            curves['R_tr'].append(R_list_tr)

            # validation
            if epoch % args.valid_interval_epochs == 0:
                L_list_va, R_list_va = run_epoch(
                    'valid', epoch, loader_va, model, None, args
                )
                curves['L_va'].append(L_list_va)
                curves['R_va'].append(R_list_va)

            # update learning rate
            scheduler.step()

        metrics['L_tr'] = np.mean(curves['L_tr'][-1])
        metrics['L_va'] = np.mean(curves['L_va'][-1])

        print('training done:', epoch+1, 'epochs')
        print('final training L:', np.mean(curves['L_tr'][-1]).item())
        print('final validation L:', np.mean(curves['L_va'][-1]).item())
        if args.mode == 'inductive':
            print('final training R:', np.mean(curves['R_tr'][-1]).item())
            print('final validation R:', np.mean(curves['R_va'][-1]).item())

    elif args.mode == 'transductive':
        print('transductive mode: nothing is done; the saved models are just random initialization')

    else:
        raise ValueError('unknown mode')

    # save model at last step
    torch.save(model.state_dict(), os.path.join(outdir, 'model_tr.pt'))

    # save training curves
    for k, v in curves.items():
        curves[k] = np.array(v)
    np.savez(os.path.join(outdir, 'curve_tr.npz'), **curves)

    # save metric
    with open(os.path.join(outdir, 'metric_tr.json'), 'w') as f:
        json.dump(metrics, f)

    # save args
    with open(os.path.join(outdir, 'args_tr.json'), 'w') as f:
        json.dump(vars(args), f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('exptname', type=str)
    parser.add_argument('trialname', type=str)
    parser.add_argument('--mode', type=str, choices=['inductive', 'transductive', 'adaptive'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--coeff-R', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, nargs=2, default=[1e-2, 1e-5])
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip-value', type=float, default=1.0)
    parser.add_argument('--grad-clip-norm', type=float, default=10.0)
    parser.add_argument('--valid-interval-epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-thT-samples-per-x', type=int, default=1)
    parser.add_argument('--maml-inner-num-iters', type=int, default=5)
    parser.add_argument('--maml-inner-learning-rate', type=float, default=1e-2)
    parser.add_argument('--maml-grad-clip-value', type=float, default=999999)

    args = parser.parse_args()

    if args.mode!='transductive' and args.coeff_R==None:
        raise ValueError('For mode!=transductive, coeff_R must be specified.')

    main(args)

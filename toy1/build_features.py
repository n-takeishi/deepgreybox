import os
import yaml
import pathlib
import numpy as np


exptname = 'toy1'


def main(args):
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_split = yaml.safe_load(fd)['split']

    rng = np.random.default_rng(args.seed)

    # load raw data
    with np.load(os.path.join(exptname, 'out', 'data_raw.npz')) as data:
        x = data['x']
        y = data['y']
        thT = data['thT']

    # shuffle
    n = x.shape[0]
    idx = rng.permutation(n)
    x = x[idx]
    y = y[idx]

    # split and save
    # x, y: (dataset_size, 1)
    n_tr = int(n*params_split['tr_ratio'])
    n_va = int(n*params_split['va_ratio'])
    outdir = os.path.join(exptname, 'out', args.trialname)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(outdir, 'data_tr.npz'),
        x=x[:n_tr,np.newaxis],
        y=y[:n_tr,np.newaxis],
        thT=np.tile(thT, (n_tr,1)))
    np.savez(os.path.join(outdir, 'data_va.npz'),
        x=x[n_tr:n_tr+n_va,np.newaxis],
        y=y[n_tr:n_tr+n_va,np.newaxis],
        thT=np.tile(thT, (n_va,1)))
    np.savez(os.path.join(outdir, 'data_te.npz'),
        x=x[n_tr+n_va:,np.newaxis],
        y=y[n_tr+n_va:,np.newaxis],
        thT=np.tile(thT, (x.shape[0]-n_tr-n_va,1)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('trialname', type=str)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    main(args)

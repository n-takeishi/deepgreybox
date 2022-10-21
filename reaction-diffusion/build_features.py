import os
import pathlib
import yaml
import numpy as np


exptname = 'reaction-diffusion'


def main(args):
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_split = yaml.safe_load(fd)['split']

    rng = np.random.default_rng(args.seed)

    # load raw data
    with np.load(os.path.join(exptname, 'out', 'data_raw.npz')) as data:
        z_all = data['UV'][:,params_split['dispose_first_steps']:] # (batch_size, len_sequence, 2, field_height, field_width)
        thT_all = data['thT']

    # shuffle
    num_episodes = z_all.shape[0]
    idx = rng.permutation(num_episodes)
    z_all = z_all[idx]
    thT_all = thT_all[idx]

    # split into tr, va, te
    n_tr = int(num_episodes*params_split['tr_ratio'])
    n_va = int(num_episodes*params_split['va_ratio'])
    z_tr = z_all[:n_tr]
    thT_tr = thT_all[:n_tr]
    z_va = z_all[n_tr:n_tr+n_va]
    thT_va = thT_all[n_tr:n_tr+n_va]
    z_te = z_all[n_tr+n_va:]
    thT_te = thT_all[n_tr+n_va:]

    # split into x and y, and save
    # x: (dataset_size, num_steps_x, 2, field_height, field_width)
    # y: (dataset_size, num_steps_y, 2, field_height, field_width)
    # x starts at t=j, y starts at t=j+1
    nsx = params_split['num_steps_x']
    nsy = params_split['num_steps_y']
    def z_to_xy(z, thT):
        N, L, _, H, W = z.shape
        x, y, thT_new = [], [], []
        for i in range(N):
            for j in range(0, L, params_split['stride']):
                if j+1+nsy > L:
                    break
                x.append(z[i,j:j+nsx].reshape(1,nsx,2,H,W))
                y.append(z[i,j+1:j+1+nsy].reshape(1,nsy,2,H,W))
                thT_new.append(thT[i][np.newaxis,:])
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0), np.concatenate(thT_new, axis=0)

    outdir = os.path.join(exptname, 'out', args.trialname)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    x, y, thT = z_to_xy(z_tr, thT_tr)
    print(x.shape, y.shape, thT.shape)
    np.savez(os.path.join(outdir, 'data_tr.npz'), x=x, y=y, thT=thT)
    x, y, thT = z_to_xy(z_va, thT_va)
    print(x.shape, y.shape, thT.shape)
    np.savez(os.path.join(outdir, 'data_va.npz'), x=x, y=y, thT=thT)
    x, y, thT = z_to_xy(z_te, thT_te)
    print(x.shape, y.shape, thT.shape)
    np.savez(os.path.join(outdir, 'data_te.npz'), x=x, y=y, thT=thT)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('trialname', type=str)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    main(args)

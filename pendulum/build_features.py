import os
import pathlib
import yaml
import numpy as np


exptname = 'pendulum'


def main(args):
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_split = yaml.safe_load(fd)['split']

    rng = np.random.default_rng(args.seed)

    # load raw data
    with np.load(os.path.join(exptname, 'out', 'data_raw.npz')) as data:
        num_episodes = data['episode_returns'].shape[0]
        z_all = data['obs'].reshape(num_episodes, -1, 3)[:,:100,:] # (N, 100, 3)
        # NOTE: latter half of each sequence is cut because by then it is usually stabilized

    # shuffle
    idx = rng.permutation(num_episodes)
    z_all = z_all[idx]

    # split into tr, va, te
    n_tr = int(num_episodes*params_split['tr_ratio'])
    n_va = int(num_episodes*params_split['va_ratio'])
    z_tr = z_all[:n_tr]
    z_va = z_all[n_tr:n_tr+n_va]
    z_te = z_all[n_tr+n_va:]

    # split into x and y, and save
    # x: (dataset_size, num_steps_x, 3)
    # y: (dataset_size, num_steps_y, 3)
    # x starts at t=j, y starts at t=j+nsx
    nsx = params_split['num_steps_x']
    nsy = params_split['num_steps_y']
    def z_to_xy(z):
        x = []
        y = []
        for i in range(z.shape[0]):
            for j in range(0, z.shape[1], params_split['stride']):
                if j+nsx+nsy > z.shape[1]:
                    break
                x.append(z[i,j:j+nsx].reshape(1,nsx,3))
                y.append(z[i,j+nsx:j+nsx+nsy].reshape(1,nsy,3))
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

    thT = np.array([[10.0]])

    outdir = os.path.join(exptname, 'out', args.trialname)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    x, y = z_to_xy(z_tr)
    print(x.shape, y.shape)
    np.savez(os.path.join(outdir, 'data_tr.npz'), x=x, y=y, thT=np.tile(thT, (x.shape[0],1)))
    x, y = z_to_xy(z_va)
    print(x.shape, y.shape)
    np.savez(os.path.join(outdir, 'data_va.npz'), x=x, y=y, thT=np.tile(thT, (x.shape[0],1)))
    x, y = z_to_xy(z_te)
    print(x.shape, y.shape)
    np.savez(os.path.join(outdir, 'data_te.npz'), x=x, y=y, thT=np.tile(thT, (x.shape[0],1)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('trialname', type=str)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    main(args)

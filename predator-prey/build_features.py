import os
import pathlib
import yaml
import numpy as np
from scipy import interpolate


exptname = 'predator-prey'


def main(args):
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_split = yaml.safe_load(fd)['split']

    rng = np.random.default_rng(args.seed)

    # load raw data
    tables_list = []
    with np.load(os.path.join(exptname, 'out', 'data_raw.npz')) as data:
        for i in range(1,11):
            if i not in params_split['exclude_tables']:
                tables_list.append(data['C%d'%i])

    # interpolation
    tables_interpolated_list = []
    for table in tables_list:
        table = table[~np.isnan(table).any(axis=1)] # eliminate rows with nan
        interp = interpolate.interp1d(table[:,0], table[:,1:], kind='linear', axis=0)
        first = np.ceil(table[0,0])
        last = np.floor(table[-1,0])
        t_interpolated = np.linspace(first, last, int(last-first)+1)
        tables_interpolated_list.append(
            np.concatenate([t_interpolated[:,np.newaxis], interp(t_interpolated)], axis=1)
        )
    tables_list = tables_interpolated_list

    # NOTE: Differently from pendulum, we divide the sequences into x and y first,
    #       and then split into train-valid-test sets. This is still a challenging setting.

    # split into x and y
    # x: (dataset_size, num_steps_x, 2)
    # y: (dataset_size, num_steps_y, 2)
    # x starts at t=j, and y starts at t=j+1
    nsx = params_split['num_steps_x']
    nsy = params_split['num_steps_y']
    def tables_to_xy(tables):
        x = []
        y = []
        for table in tables:
            for j in range(0, table.shape[0], params_split['stride']):
                if j+1+nsy > table.shape[0]:
                    break
                x_cand = table[j:j+nsx,1:3].reshape(1,nsx,2)
                y_cand = table[j+1:j+1+nsy,1:3].reshape(1,nsy,2)
                if np.max(x_cand) > params_split['max_value'] \
                    or np.max(y_cand) > params_split['max_value']:
                    continue
                x.append(x_cand.copy())
                y.append(y_cand.copy())
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

    x_all, y_all = tables_to_xy(tables_list)

    # shuffle
    idx = rng.permutation(x_all.shape[0])
    x_all = x_all[idx]
    y_all = y_all[idx]

    # split into tr, va, te
    n_tr = int(x_all.shape[0]*params_split['tr_ratio'])
    n_va = int(x_all.shape[0]*params_split['va_ratio'])
    x_tr = x_all[:n_tr]; y_tr = y_all[:n_tr]
    x_va = x_all[n_tr:n_tr+n_va]; y_va = y_all[n_tr:n_tr+n_va]
    x_te = x_all[n_tr+n_va:]; y_te = y_all[n_tr+n_va:]

    # save
    print('max:', np.max(y_all), 'min:', np.min(y_all))
    print()
    print(x_tr.shape, y_tr.shape)
    outdir = os.path.join(exptname, 'out', args.trialname)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(outdir, 'data_tr.npz'), x=x_tr, y=y_tr)
    print(x_va.shape, y_va.shape)
    np.savez(os.path.join(outdir, 'data_va.npz'), x=x_va, y=y_va)
    print(x_te.shape, y_te.shape)
    np.savez(os.path.join(outdir, 'data_te.npz'), x=x_te, y=y_te)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('trialname', type=str)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    main(args)

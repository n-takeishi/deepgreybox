import os
import sys
import pathlib
import yaml
import numpy as np


exptname = 'toy1'
seed = 0


def data_generating_process(a, b, x):
    yT = a*np.sin(x)
    y = yT + b*np.cos(x)
    return y, yT


def main():
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_data = yaml.safe_load(fd)['data']

    rng = np.random.default_rng(seed)

    x = rng.uniform(-np.pi, np.pi, params_data['num_samples'])
    y, _ = data_generating_process(params_data['a'], params_data['b'], x)

    # observation noise
    y = y + rng.standard_normal(y.shape)*params_data['sigma']

    # ground truth of thT (1, dim_thT)
    thT = np.array([[params_data['a'], 0.0]])

    # save
    outdir = os.path.join(exptname, 'out')
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(outdir, 'data_raw.npz'), x=x, y=y, thT=thT)
    print('saved', x.shape[0], 'instances of x & y')


if __name__ == '__main__':
    main()

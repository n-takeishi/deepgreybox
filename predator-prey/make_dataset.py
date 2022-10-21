import os
import pathlib
import yaml
import urllib.request
import shutil
import zipfile
import tempfile
import numpy as np


exptname = 'predator-prey'
seed = 0


def main():
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_data = yaml.safe_load(fd)['data']

    tables_dict = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download zip file
        zip_path = os.path.join(tmpdirname, 'predatory-prey_original.zip')
        with urllib.request.urlopen(params_data['url']) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        # unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(tmpdirname))

        # load csv files
        for i in range(1,11):
            tables_dict['C%d'%i] = np.genfromtxt(os.path.join(tmpdirname, 'C%d.csv'%i), skip_header=1, delimiter=',')

            # take only first three columns
            tables_dict['C%d'%i] = tables_dict['C%d'%i][:,0:3]

            # normalize manually
            tables_dict['C%d'%i][:,1] *= params_data['normalize_coeff'][0]
            tables_dict['C%d'%i][:,2] *= params_data['normalize_coeff'][1]

    # save data in npz
    outdir = os.path.join(exptname, 'out')
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(outdir, 'data_raw.npz'), **tables_dict)
    print('saved raw data')

if __name__ == '__main__':
    main()

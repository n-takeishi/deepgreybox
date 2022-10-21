import os
import pathlib
import yaml
import urllib.request
import shutil


exptname = 'pendulum'
seed = 0


def main():
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_data = yaml.safe_load(fd)['data']

    # download and save
    outdir = os.path.join(exptname, 'out')
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(params_data['url']) as response, \
        open(os.path.join(outdir, 'data_raw.npz'), 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


if __name__ == '__main__':
    main()
import os
import pathlib
import yaml
import numpy as np
import torch
import torchdiffeq


exptname = 'reaction-diffusion'
seed = 0


def laplacian(field, mesh_step):
    '''
    field: (batch_size, field_size, field_size)
    '''

    _field = field.clone()

    # expand the field following Neumann boundary condition
    pad_top = _field[:,0,:].unsqueeze(1)
    pad_bottom = _field[:,-1,:].unsqueeze(1)
    _field = torch.cat([pad_top, _field, pad_bottom], dim=1)
    pad_left = _field[:,:,0].unsqueeze(2)
    pad_right = _field[:,:,-1].unsqueeze(2)
    _field = torch.cat([pad_left, _field, pad_right], dim=2)

    # compute Laplacian by five-point stencil
    top = _field[:, :-2, 1:-1]
    left = _field[:, 1:-1, :-2]
    bottom = _field[:, 2:, 1:-1]
    right = _field[:, 1:-1, 2:]
    center = _field[:, 1:-1, 1:-1]
    return (top + left + bottom + right - 4.0*center) / mesh_step**2


def main():
    with open(os.path.join(exptname, 'params.yaml'), 'r') as fd:
        params_data = yaml.safe_load(fd)['data']

    device = torch.device(params_data['device'])

    torch.manual_seed(seed)

    with torch.no_grad():
        a = torch.rand(1, 1, device=device).expand(params_data['num_episodes'],1) \
            * (params_data['a_range'][1]-params_data['a_range'][0]) + params_data['a_range'][0]
        b = torch.rand(1, 1, device=device).expand(params_data['num_episodes'],1) \
            * (params_data['b_range'][1]-params_data['b_range'][0]) + params_data['b_range'][0]
        k = params_data['k']

        def fun(t, state):
            '''
            t: scalar
            state: (batch_size, 2, field_size, field_size)
            '''

            U, V = state[:,0], state[:,1]
            deltaU = laplacian(U, params_data['mesh_step']) # (batch_size, field_size, field_size)
            deltaV = laplacian(V, params_data['mesh_step'])
            dU = a.view(-1,1,1)*deltaU + U - U**3 - k - V
            dV = b.view(-1,1,1)*deltaV + U - V
            out = torch.cat([dU.unsqueeze(1), dV.unsqueeze(1)], dim=1)
            return out

        t_eval = torch.linspace(params_data['t_span'][0], params_data['t_span'][1], params_data['len_sequence'], device=device)
        init_cond = torch.rand(params_data['num_episodes'], 2, params_data['field_size'], params_data['field_size'], device=device)
        UV = torchdiffeq.odeint(
            fun,
            init_cond,
            t_eval,
            method = 'rk4',
            options = {'step_size': 1e-3}
        ) # (len_sequence, batch_size, 2, field_size, field_size)
        UV = UV.permute(1,0,2,3,4) # (batch_size, len_sequence, 2, field_size, field_size)

        # noise
        UV += torch.randn_like(UV)*params_data['sigma']

        UV = UV.cpu().numpy()
        thT = torch.cat([a,b], dim=1).cpu().numpy()

    # save
    outdir = os.path.join(exptname, 'out')
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(outdir, 'data_raw.npz'), UV=UV, thT=thT)
    print('saved data of shape', UV.shape)


if __name__ == '__main__':
    main()

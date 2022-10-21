import torch
from torch import nn
import torchdiffeq

class GreyboxModel(nn.Module):
    def __init__(self, params_model):
        super().__init__()

        self.thT_bound = params_model['thT_bound']
        self.params_odeint = params_model['odeint']
        self.field_size = params_model['field_size']

        # t at which ODE solution is evaluated
        self.register_buffer('t_eval',
            torch.linspace(
                0.0,
                self.params_odeint['dt']*params_model['num_steps_y'],
                params_model['num_steps_y']+1
            )
        )

        use_batchnorm = params_model['batchnorm']['use_batchnorm']
        track_running_stats = params_model['batchnorm']['track_running_stats']

        # fD
        k = params_model['fD']['sizes_hidden_layers']
        net_fD_mlp = []
        net_fD_mlp.append(nn.Linear(2, k[0]))
        net_fD_mlp.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_fD_mlp.append(nn.Linear(k[i], k[i+1]))
            net_fD_mlp.append(nn.LeakyReLU(inplace=True))
        net_fD_mlp.append(nn.Linear(k[-1], 2)) # 2*2*3*3
        self.net_fD_mlp = nn.Sequential(*net_fD_mlp)

        k = params_model['fD']['nums_hidden_channels']
        net_fD_conv = []
        net_fD_conv.append(nn.Conv2d(2, k[0], 3, stride=1, padding=1, padding_mode='replicate', bias=False))
        if use_batchnorm:
            net_fD_conv.append(nn.BatchNorm2d(k[0], track_running_stats=track_running_stats))
        net_fD_conv.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_fD_conv.append(nn.Conv2d(k[i], k[i+1], 3, stride=1, padding=1, padding_mode='replicate', bias=False))
            if use_batchnorm:
                net_fD_conv.append(nn.BatchNorm2d(k[i+1], track_running_stats=track_running_stats))
            net_fD_conv.append(nn.LeakyReLU(inplace=True))
        net_fD_conv.append(nn.Conv2d(k[-1], 2, 3, stride=1, padding=1, padding_mode='replicate'))
        self.net_fD_conv = nn.Sequential(*net_fD_conv)

        # thT_fixed
        thT_init = self.draw_thT()
        self.prm_thT_fixed = nn.Parameter(thT_init)

    def draw_thT(self, num_samples=1, device='cpu'):
        out = torch.empty(num_samples, len(self.thT_bound), device=device)
        for j, bound in enumerate(self.thT_bound):
            nn.init.uniform_(out[:,j], a=bound[0], b=bound[1])
        return out

    def thT_encode(self, x):
        N, _, _, H, W = x.shape
        tmp = self.net_thT_encoder(x.view(N, -1, H, W)) # (N,2,H,W)
        tmp = torch.mean(torch.sigmoid(tmp), dim=[2,3]) # (N,2)
        out = torch.empty_like(tmp)
        for j,bound in enumerate(self.thT_bound):
            out[:,j] = tmp[:,j]*(bound[1] - bound[0]) + bound[0]
        return out

    def laplacian(self, field):
        # field: (batch_size, field_size, field_size)
        ## TODO: use F.pad and F.conv2d

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
        return (top + left + bottom + right - 4.0*center) / self.params_odeint['mesh_step']**2

    def fT(self, state, thT):
        # state: (batch_size, 2, field_size, field_size)
        # thT: (batch_size, 2) or something broadcastable to it

        a, b = thT[:,0].view(-1,1,1), thT[:,1].view(-1,1,1)
        U, V = state[:,0], state[:,1]
        deltaU = self.laplacian(U)
        deltaV = self.laplacian(V)
        dU = a*deltaU
        dV = b*deltaV
        return torch.cat([dU.unsqueeze(1), dV.unsqueeze(1)], dim=1)

    def fD(self, state, thT, fTs):
        # state: (batch_size, 2, field_size, field_size)
        # thT: (batch_size, 2)
        # fTs: (batch_size, 2, field_size, field_size)
        batch_size = state.shape[0]
        field_size = self.field_size

        U, V = state[:,0], state[:,1]
        deltaU = self.laplacian(U)
        deltaV = self.laplacian(V)
        weights = torch.tanh(self.net_fD_mlp(100.0*thT.expand(batch_size,-1)))*0.01 # (N,2)
        weights = torch.clamp(weights, min = -0.5*thT)
        tmp = torch.cat([
            (weights[:,0].view(-1,1,1)*deltaU).unsqueeze(1),
            (weights[:,1].view(-1,1,1)*deltaV).unsqueeze(1)
        ], dim=1)

        return self.net_fD_conv(state) + tmp

    def forward(self, x, num_thT_samples_per_x=0, thT=None, return_all_R=False):
        # x: (batch_size, num_steps_x, 2, field_size, field_size)
        # thT: (batch_size, 2) or something broadcastable to it
        batch_size, num_steps_x, _, _, _ = x.shape

        if num_thT_samples_per_x > 0:
            thT = self.draw_thT(batch_size*num_thT_samples_per_x, device=x.device)
            _x = x.repeat(num_thT_samples_per_x, 1, 1, 1, 1)
        else:
            if thT is None:
                thT = self.prm_thT_fixed
            _x = x

        compT_list, compD_list = [], []
        def fun(t, state):
            fTs = self.fT(state, thT)
            fDs = self.fD(state, thT, fTs)
            compT_list.append(fTs)
            compD_list.append(fDs)
            return fTs + fDs

        # set initial condition = first snapshot of x
        init_cond = _x[:,0] # (batch_size, 2, field_size, field_size)

        # solve ODE
        method = self.params_odeint['method']
        yh = torchdiffeq.odeint(
            fun,
            init_cond,
            self.t_eval,
            method = method,
            options = self.params_odeint[method] if method in self.params_odeint else None
        ) # (num_steps_y+1, batch_size, 2, field_size, field_size)
        yh = yh[1:].permute(1,0,2,3,4) # (batch_size, num_steps_y, 2, field_size, field_size)

        compT = torch.cat(compT_list, dim=0)
        compD = torch.cat(compD_list, dim=0)
        normD =  torch.pow(compD, 2).sum(dim=[1,2,3]).mean()
        abs_dotTD = torch.abs((compT*compD).sum(dim=[1,2,3]).mean())
        R = torch.log(normD) + torch.log(abs_dotTD)

        if return_all_R:
            return yh, normD, abs_dotTD
        else:
            return yh, R


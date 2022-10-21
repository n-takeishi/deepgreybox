import torch
from torch import nn
import torchdiffeq

class GreyboxModel(nn.Module):
    def __init__(self, params_model):
        super().__init__()

        self.thT_bound = params_model['thT_bound']
        self.params_odeint = params_model['odeint']

        # t at which ODE solution is evaluated
        self.register_buffer('t_eval',
            torch.linspace(
                0.0,
                self.params_odeint['dt']*params_model['num_steps_y'],
                params_model['num_steps_y']+1
            )
        )

        # fD
        k = params_model['fD']['sizes_hidden_layers']
        net_fD = []
        net_fD.append(nn.Linear(6, k[0]))
        net_fD.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_fD.append(nn.Linear(k[i], k[i+1]))
            net_fD.append(nn.LeakyReLU(inplace=True))
        net_fD.append(nn.Linear(k[-1], 2))
        self.net_fD = nn.Sequential(*net_fD)

        # thT_fixed
        thT_init = self.draw_thT()
        self.prm_thT_fixed = nn.Parameter(thT_init)

    def draw_thT(self, num_samples=1, device='cpu'):
        out = torch.empty(num_samples, len(self.thT_bound), device=device)
        for j, bound in enumerate(self.thT_bound):
            nn.init.uniform_(out[:,j], a=bound[0], b=bound[1])
        return out

    def fT(self, state, thT):
        # state: (batch_size, 2)
        # thT: (batch_size, 1) or something broadcastable to it

        g = thT[:,0].unsqueeze(1); l = 1.0
        angle, angvel = state[:,0].unsqueeze(1), state[:,1].unsqueeze(1)
        d_angle = angvel
        d_angvel = 3.0*g / (2.0*l) * torch.sin(angle)
        return torch.cat([d_angle, d_angvel], dim=1)

    def fD(self, state, thT, fTs):
        # state: (batch_size, 2)
        # thT: (batch_size, 1) or something broadcastable to it
        # fTs: (batch_size, 2)
        tmp = torch.cat([
            torch.cos(state[:,0]).view(-1,1),
            torch.sin(state[:,0]).view(-1,1),
            state[:,1].view(-1,1),
            thT.expand(state.shape[0],-1),
            fTs
        ], dim=1)
        return self.net_fD(tmp)

    def forward(self, x, num_thT_samples_per_x=0, thT=None, return_all_R=False):
        # x: (batch_size, num_steps_x, 3)
        # usually num_steps_x = 1, last dim = cos(angle) & sin(angle) & angular vel
        batch_size, num_steps_x, _ = x.shape

        if num_thT_samples_per_x > 0:
            thT = self.draw_thT(batch_size*num_thT_samples_per_x, device=x.device)
            _x = x.repeat(num_thT_samples_per_x, 1, 1)
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

        # set initial condition = last snapshot of x
        cos_angle_x, sin_angle_x, angvel_x = _x[:,:,0], _x[:,:,1], _x[:,:,2]
        angle_x = torch.atan2(sin_angle_x, cos_angle_x)
        init_cond = torch.cat([
            angle_x[:,-1].unsqueeze(1),
            angvel_x[:,-1].unsqueeze(1)], dim=1) # (batch_size, 2)

        # solve ODE
        method = self.params_odeint['method']
        yh = torchdiffeq.odeint(
            fun,
            init_cond,
            self.t_eval,
            method = method,
            options = self.params_odeint[method] if method in self.params_odeint else None
        ) # (num_steps_y+1, batch_size, 2)
        yh = yh[1:].permute(1,0,2) # (batch_size, num_steps_y, 2)

        # convert to feature (angle --> cos(angle) & sin(angle))
        yh = torch.cat([
            torch.cos(yh[:,:,0]).view(-1,yh.shape[1],1),
            torch.sin(yh[:,:,0]).view(-1,yh.shape[1],1),
            yh[:,:,1].view(-1,yh.shape[1],1)
        ], dim=2) # (batch_size, num_steps_y, 3)

        compT = torch.cat(compT_list, dim=0)
        compD = torch.cat(compD_list, dim=0)
        normT =  torch.pow(compT, 2).sum(dim=1).mean()
        normD =  torch.pow(compD, 2).sum(dim=1).mean()
        abs_dotTD = torch.abs((compT*compD).sum(dim=1).mean())
        R = abs_dotTD

        if return_all_R:
            return yh, normT, normD, abs_dotTD
        else:
            return yh, R

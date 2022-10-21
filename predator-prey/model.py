import torch
from torch import nn
import torchdiffeq


class suppress_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)


class GreyboxModel(nn.Module):
    def __init__(self, params_model):
        super().__init__()

        self.thT_bound = params_model['thT_bound']
        self.params_odeint = params_model['odeint']
        self.depth_t_embed = params_model['depth_t_embed']

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
        net_fD.append(nn.Linear(2*self.depth_t_embed+8+params_model['zD_encoder']['dim_out'], k[0]))
        net_fD.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_fD.append(nn.Linear(k[i], k[i+1]))
            net_fD.append(nn.LeakyReLU(inplace=True))
        net_fD.append(nn.Linear(k[-1], 2))
        self.net_fD = nn.Sequential(*net_fD)

        # zD_encoder
        k = params_model['zD_encoder']['sizes_hidden_layers']
        net_zD_encoder = []
        net_zD_encoder.append(nn.Linear(params_model['num_steps_x']*2, k[0]))
        net_zD_encoder.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_zD_encoder.append(nn.Linear(k[i], k[i+1]))
            net_zD_encoder.append(nn.LeakyReLU(inplace=True))
        net_zD_encoder.append(nn.Linear(k[-1], params_model['zD_encoder']['dim_out']))
        self.net_zD_encoder = nn.Sequential(*net_zD_encoder)

        # thT_encoder
        k = params_model['thT_encoder']['sizes_hidden_layers']
        net_thT_encoder = []
        net_thT_encoder.append(nn.Linear(params_model['num_steps_x']*2, k[0]))
        net_thT_encoder.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_thT_encoder.append(nn.Linear(k[i], k[i+1]))
            net_thT_encoder.append(nn.LeakyReLU(inplace=True))
        net_thT_encoder.append(nn.Linear(k[-1], 4))
        self.net_thT_encoder = nn.Sequential(*net_thT_encoder)

    def draw_thT(self, num_samples=1, device='cpu'):
        out = torch.empty(num_samples, len(self.thT_bound), device=device)
        for j, bound in enumerate(self.thT_bound):
            nn.init.uniform_(out[:,j], a=bound[0], b=bound[1])
        return out

    def thT_encode(self, x):
        tmp = torch.sigmoid(self.net_thT_encoder(x.view(x.shape[0], -1)))
        out = torch.empty_like(tmp)
        for j,bound in enumerate(self.thT_bound):
            out[:,j] = tmp[:,j]*(bound[1] - bound[0]) + bound[0]
        return out

    def zD_encode(self, x):
        return self.net_zD_encoder(x.view(x.shape[0], -1))

    def fT(self, state, thT):
        # state: (batch_size, 2), last dim = prey & predator

        b, d, p, r = [thT[:,i].unsqueeze(1) for i in range(4)]
        prey, predator = state[:,0].unsqueeze(1), state[:,1].unsqueeze(1)
        d_prey = prey * (b - p*predator)
        d_predator = predator * (r*prey - d)
        return torch.cat([d_prey, d_predator], dim=1)

    def fD(self, t, state, zD, thT, fTs):
        # t: scalar
        # state: (batch_size, 2)
        # zD: (batch_size, dim_zD)
        # thT: (batch_size, 4) or something broadcastable to it
        # fTs: (batch_size, 2)

        t_embed = self.get_t_embed(t.expand(state.shape[0],1))

        tmp = torch.cat([t_embed, state, zD, thT.expand(state.shape[0],-1), fTs], dim=1)
        return self.net_fD(tmp)

    def get_t_embed(self, t):
        # t: (batch_size, 1)
        t_max = self.t_eval[-1]
        t_embed = []
        for i in range(self.depth_t_embed):
            t_embed.append(torch.sin((i+1)*6.2831853*t/t_max))
            t_embed.append(torch.cos((i+1)*6.2831853*t/t_max))
        return torch.cat(t_embed, dim=1)

    def forward(self, x,  num_thT_samples_per_x=0, t_eval=None, thT=None):
        # x: (batch_size, num_steps_x, 2); last dim = prey & predator
        # thT: (batch_size, 2) or something broadcastable to it
        batch_size, num_steps_x, _ = x.shape

        zD = self.zD_encode(x)

        if num_thT_samples_per_x > 0:
            thT = self.draw_thT(batch_size*num_thT_samples_per_x, device=x.device)
            _x = x.repeat(num_thT_samples_per_x, 1, 1)
            _zD = zD.repeat(num_thT_samples_per_x, 1)
        else:
            if thT is None:
                thT = self.thT_encode(x)
            _x = x
            _zD = zD

        compT_list, compD_list = [], []
        def fun(t, state):
            _state = torch.clamp(state, min=0.0, max=self.params_odeint['max_state_value'])
            fTs = self.fT(_state, thT)
            fDs = self.fD(t, _state, _zD, thT, fTs)
            compT_list.append(fTs)
            compD_list.append(fDs)
            out = fTs + fDs
            if self.params_odeint['suppress_grad']:
                out = suppress_grad.apply(out)
            return out

        # set initial condition = first snapshot of x
        init_cond = _x[:,0,:]

        # solve ODE
        method = self.params_odeint['method']
        yh = torchdiffeq.odeint(
            fun,
            init_cond,
            self.t_eval if t_eval==None else t_eval,
            method = method,
            options = self.params_odeint[method] if method in self.params_odeint else None
        ) # (num_steps_y+1, batch_size, 2)
        if t_eval==None:
            yh = yh[1:]
        yh = yh.permute(1,0,2) # (batch_size, num_steps_y, 2)

        # compT = torch.cat(compT_list, dim=0)
        compD = torch.cat(compD_list, dim=0)
        normD =  torch.pow(compD, 2).sum(dim=1).mean()
        # abs_dotTD = torch.abs((compT*compD).sum(dim=1).mean())
        R = normD

        return yh, R
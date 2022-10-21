import torch
from torch import nn

class GreyboxModel(nn.Module):
    def __init__(self, params_model):
        super().__init__()

        self.thT_bound = params_model['thT_bound']

        # fD
        k = params_model['fD']['sizes_hidden_layers']
        net_fD = []
        net_fD.append(nn.Linear(4, k[0]))
        net_fD.append(nn.LeakyReLU(inplace=True))
        for i in range(len(k)-1):
            net_fD.append(nn.Linear(k[i], k[i+1]))
            net_fD.append(nn.LeakyReLU(inplace=True))
        net_fD.append(nn.Linear(k[-1], 1))
        self.net_fD = nn.Sequential(*net_fD)

        # thT_fixed
        thT_init = self.draw_thT()
        self.prm_thT_fixed = nn.Parameter(thT_init)

    def draw_thT(self, num_samples=1, device='cpu'):
        out = torch.empty(num_samples, len(self.thT_bound), device=device)
        for j, bound in enumerate(self.thT_bound):
            nn.init.uniform_(out[:,j], a=bound[0], b=bound[1])
        return out

    def fT(self, x, thT):
        # x: (batch_size, 1)
        # thT: (batch_size, 2) or something broadcastable to it
        a, c = thT[:,0].unsqueeze(1), thT[:,1].unsqueeze(1)
        return a * torch.sin(x + c)

    def fD(self, x, thT, fTx):
        # x: (batch_size, 1)
        # thT: (batch_size, 2) or something broadcastable to it
        # fTx: (batch_size, 1)
        return self.net_fD(torch.cat([x, thT.expand(x.shape[0],-1), fTx], dim=1))

    def forward(self, x, num_thT_samples_per_x=0, thT=None, return_all_R=False):
        # x: (batch_size, 1)
        batch_size, _ = x.shape

        if num_thT_samples_per_x > 0:
            thT = self.draw_thT(batch_size*num_thT_samples_per_x)
            _x = x.repeat(num_thT_samples_per_x, 1)
        else:
            if thT is None:
                thT = self.prm_thT_fixed
            _x = x

        fTx = self.fT(_x, thT)
        fDx = self.fD(_x, thT, fTx)
        compT = fTx
        compD = fDx
        yh = fTx + fDx

        normT = torch.pow(compT, 2).mean()
        normD = torch.pow(compD, 2).mean()
        abs_dotTD = torch.abs((compT * compD).mean())
        abs_diff_normTD = torch.abs(normT - normD)
        sq_c = torch.pow(self.prm_thT_fixed[:,1], 2).mean()
        R = abs_dotTD + abs_diff_normTD + sq_c

        if return_all_R:
            return yh, normT, normD, abs_dotTD, abs_diff_normTD, sq_c
        else:
            return yh, R

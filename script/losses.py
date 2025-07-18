import torch
from torch import nn


class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    targets # [N, 3]
    inputs['rgb_coarse'] # [N, 3]
    inputs['rgb_fine'] # [N, 3]
    inputs['beta'] # [N]
    inputs['transient_sigmas'] # [N, 2*N_Samples]
    :return:
    """

    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, depth_targets):
        ret = {}
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse'] - targets) ** 2).mean()

        if depth_targets != None:
            depth_targets = depth_targets.squeeze()
            mask = depth_targets > 0
            valid_depth_targets = depth_targets[mask]

            # loss for depth
            ret['d_l'] = 0.09 * (abs(inputs['render_depth'][mask] - valid_depth_targets) / torch.sqrt(inputs['render_depth_var'][mask])).mean()
            # ret['d_l'] = 0.1 * ((inputs['render_depth'][mask] - valid_depth_targets) ** 2).mean()  # [0.1, 0.04]
            # ret['d_l'] =0.1 * (inputs['render_depth'][mask] - valid_depth_targets).abs().mean()

        if 'rgb_fine' in inputs:
            if 'beta' not in inputs:  # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((inputs['rgb_fine'] - targets) ** 2).mean()
            else:
                ret['f_l'] = ((inputs['rgb_fine'] - targets) ** 2 / (2 * inputs['beta'].unsqueeze(1) ** 2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean()  # +3 to make it positive
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret


loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss}
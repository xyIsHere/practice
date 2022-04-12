import torch
import torch.nn as nn


def batch_norm(x, moving_mean, moving_var, gamma, beta, momentum=0.1, eps=1e-5):
    # for inference
    if not torch.is_grad_enabled():
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)

    else:
        assert len(x.shape) in (2, 4), "wrong dim"
        # fully connected
        if len(x.shape) == 2:
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        # convolution
        else:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # with keepdim [1,16,1,1];without [16]
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        ｘ_hat = (x - mean) / torch.sqrt(var + eps)

        # update the moving_mean and moving_var
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    y = gamma * x_hat + beta

    return y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dim):
        super(BatchNorm, self).__init__()
        assert len(x.shape) in (2, 4), "wrong dim"
        if num_dim == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        y, self.moving_mean, self.moving_var = batch_norm(x, self.moving_mean, self.moving_var, self.gamma, self.beta)

        return y


if __name__ == '__main__':
    x = torch.randn((2, 16, 128, 128))
    bn = BatchNorm(16, 4)
    out = bn(x)
    print(out.shape)






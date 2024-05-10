import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, args):
        super(Extractor, self).__init__()
        self.args = args
        self.linear = nn.Linear(args.seq_len, args.seq_len)
        self.activation = nn.ReLU()
    
    def latent_mixup(self, batch_x_a, alpha, is_classification=False):
        N, T, F = batch_x_a.shape 
        # lamb = np.random.beta(alpha, alpha) 
        lamb = torch.distributions.Beta(20, 0.5).sample((N, 1, 1)).to(batch_x_a.device)
        lamb = torch.clip(lamb, 0.5, 1.0)
        idx = torch.randperm(N).cuda()
        batch_x_b = batch_x_a[idx].clone()
        # mse:0.395765095949173, mae:0.4104437232017517
        mix_batch = lamb*batch_x_a + (1-lamb)*batch_x_b
        return mix_batch, lamb, idx

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x)  # Apply the linear transformation
        x = self.activation(x)  # Apply the activation function
        x = x.permute(0, 2, 1)
        with torch.no_grad():
            x = self.latent_mixup(x, self.args.alpha)
        return x
import torch
import numpy as np
import torch.fft as fft

def none_aug(x):
    return x


def jitter(x, sigma=0.03):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def hflip(x):    
    return x[::-1, :]


def vflip(x):
    x = x.reshape((1, x.shape[0], x.shape[1]))    
    mean_per_feature = x.mean(axis=1, keepdims=True)
    x = 2 * mean_per_feature - x
    return x[0]


def scaling(x, sigma=0.1):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])[0]


def window_warp(x, window_ratio=0.3, scales=[0.5, 2.0]):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret[0]


def window_slice(x, reduce_ratio=0.9):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret[0]


def permutation(x, max_segments=5, seg_mode="equal"):
    x = x.reshape((1, x.shape[0], x.shape[1]))
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret[0]


# # # Original Mixup
def orig_mixup(batch_x_a, label_a, alpha, emixup_vflip_rate, is_classification=False):
    N, T, F = batch_x_a.shape 
    lamb = np.random.beta(alpha, alpha) 
    idx = torch.randperm(N).cuda()
    batch_x_b = batch_x_a[idx].clone()
    mix_batch = lamb*batch_x_a + (1-lamb)*batch_x_b
    return mix_batch, label_a, lamb, idx

# Function to flip the tensor values vertically around the mean for each feature
def flip_vertical_around_mean(data):
    mean_per_feature = data.mean(dim=1, keepdim=True)  # Mean across time, keeping dimensions
    return 2 * mean_per_feature - data


def GNM(batch_x_a, scale=0.01):
    N, T, F = batch_x_a.shape 
    std_a = batch_x_a.std(dim=1, keepdim=True)
    mean_std_a = torch.mean(std_a, dim=0, keepdim=True)
    noise_a = torch.zeros_like(batch_x_a)
    for i in range(F):
        noise_a[:,:,i] = torch.normal(mean=0., std=scale*mean_std_a[0,0,i], size=[1, T], device=batch_x_a.device)
    batch_x_a = batch_x_a + noise_a
    return batch_x_a


def enahced_mixup(batch_x_a, label_a, alpha, emixup_vflip_rate, is_classification=False):
    N, T, F = batch_x_a.shape 
    lamb = torch.distributions.Beta(40, 0.5).sample((1, 1, 1)).to(batch_x_a.device)
    idx = torch.randperm(N).to(batch_x_a.device)
    # Step 1
    if torch.rand([]) < emixup_vflip_rate:
        batch_x_a = flip_vertical_around_mean(batch_x_a)
        if not is_classification:
            label_a = flip_vertical_around_mean(label_a)

    # Step 2    
    batch_x_b = batch_x_a[idx].clone()
    batch_x_a = GNM(batch_x_a, scale=0.01)
    batch_x_b = GNM(batch_x_b, scale=0.01)
    
    # Step 3
    mix_batch = lamb*batch_x_a + (1-lamb)*batch_x_b

    # Step 4    
    mix_batch = GNM(mix_batch, scale=0.01)
    
    return mix_batch, label_a, lamb, idx

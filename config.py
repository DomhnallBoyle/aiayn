import torch

d_model = 512  # model dims
d_ff = 2048  # ffn dims
num_heads = 8  # num attn heads
d_k = 64  # key dims
d_v = 64  # value dims
max_len = 5000
num_layers = 6
label_smoothing = 0.1
warmup_steps = 4000
lr_initial = 1e-3
lr_betas = (0.9, 0.98)
lr_eps = 1e-9
p_drop = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

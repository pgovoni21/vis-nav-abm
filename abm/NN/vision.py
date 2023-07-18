import torch
import torch.nn as nn

class Stem(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=2, stride=2),
            # LayerNorm(dim_out, eps=1e-6, data_format="channels_first")  ## left out from OG - better speed+accuracy
        )
    def forward(self, x):
        return self.stem(x)

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.downsample_layer = nn.Sequential(
                # LayerNorm(dim_in, eps=1e-6, data_format="channels_first"), ## OG - did not train (channel-wise local)
                nn.GroupNorm(num_groups=1, num_channels=dim_in),             ## switched from OG (spatial/channel global)
                nn.Conv1d(dim_in, dim_out, kernel_size=2, stride=2),
            )
    def forward(self, x):
        return self.downsample_layer(x)

# ------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim, activ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers

        # set activation function
        if activ == 'relu': self.activ = torch.relu
        elif activ == 'tanh': self.activ = torch.tanh
        elif activ == 'silu': self.activ = torch.nn.SiLU()
        elif activ == 'gelu': self.active = torch.nn.GELU()
        else: raise ValueError(f'Invalid activation function: {activ}')

        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, W) -> (N, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activ(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, W, C) -> (N, C, W)
        x = x + input
        return x

class Stages(nn.Module):
    def __init__(self, dim, dep, activ):
        super().__init__()
        self.stages = nn.Sequential(
                *[Block(dim, activ) for _ in range(dep)]
            )
    def forward(self, x):
        return self.stages(x)

# ------------------------------------------------------------

class ConvNeXt(nn.Module):
    def __init__(self, 
                 in_dims=6, 
                 depths=[1, 2, 1], 
                 dims=[2, 4, 6], 
                 activ='gelu',):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Stem(in_dims, dims[0]))
        self.layers.append(Stages(dims[0], depths[0], activ))

        self.num_layers_past_stem = len(dims)-1
        for i in range(self.num_layers_past_stem):

            self.layers.append(Downsample(dims[i], dims[i+1]))
            self.layers.append(Stages(dims[i+1], depths[i+1], activ))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(layer, layer.parameters)
            # print(f'-----> {x}')
        x = self.norm(x.mean([-1]))   # global average pooling, (N, C, H, W) -> (N, C)
        return x

# ------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            x = x.permute(0, 2, 1)
            x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 2, 1)
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)     # sum of squares across spatial dim
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)  # divide by mean to normalize
        return self.weight * (x * Nx) + self.bias + x

# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # image = torch.rand(1, 2, 3)
    # print(f'Image: {image.shape}')

    model = ConvNeXt(
        in_dims=2,
        depths=[1,], 
        dims=[4,],
        activ='silu',
        )
    # print(f'Model: {model(image).shape}') # torch.Size([1, {outputs}])

    # model(image)
    print(sum(p.numel() for p in model.parameters()))

    # for m in model.modules():
    #     if isinstance(m, (nn.Linear, nn.Conv1d, nn.LayerNorm, LayerNorm, GRN)):
            
    #         print(m)
    #         params = sum(p.numel() for p in m.parameters())
    #         print(params)

    #         for p in m.parameters():
    #             print(p,p.shape)
            
    #             p.data = torch.zeros_like(p)

    #         print()
    

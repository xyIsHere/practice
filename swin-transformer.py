import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape # B是1  C3 H224  W224
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C  1 3136 96  3136就是我们的56*56 压扁
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=(56,56), dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution #
        B, L, C = x.shape ## 输入进来x为1 3136 96
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C) ## 这里x变为了 1 56 56  96

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C ## x0形状为1 28 28 96
        print(x0.shape)
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C ## x1形状为1 28 28 96
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C## ## x2形状为1 28 28 96
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C ## ## x3形状为1 28 28 96
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  ## x为1 28 28 384
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C ## 1 784 384

        x = self.norm(x)
        x = self.reduction(x) # 1 784 192

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


if __name__ == "__main__":
    img = torch.zeros(1, 3, 224, 224)
    patchEmded = PatchEmbed()
    patchMerge = PatchMerging()
    out = patchEmded(img)
    out = patchMerge(out)
    print(out.shape)
    # vit = ViT(image_size=256, patch_size=16, image_channel=3, pool='cls', dim=1024)
    # out = vit(img)
    #
    # transblock = TransformerBlock(embed_dim=512)
    # out = transblock(out)
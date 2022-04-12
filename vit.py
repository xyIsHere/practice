import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange


# helper
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head=8, head_dim=64, dropout=0.1):
        super().__init__()
        inner_dim = head * head_dim
        project_out = not(head == 1 and head_dim == dim)

        self.head = head
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_qkv_rearrangefn = Rearrange('b n (h d) -> b h n d', h=self.head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_orifn = Rearrange('b h n d -> b n (h d)', h=self.head)

        self.to_out =  nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # obtain qkv via three different linear layers
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: self.to_qkv_rearrangefn(t), qkv)

        q_k = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(q_k)
        attn = self.dropout(attn)

        # multiply with v
        out = torch.matmul(attn, v)
        out = self.to_orifn(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head=8, head_dim=64, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.prenorm1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttention(embed_dim, head, head_dim, dropout)
        self.prenorm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        ori = x
        x = self.multi_head_attention(x)
        x = self.prenorm1(x) + ori

        ori = x
        x = self.feed_forward(x)
        x = self.prenorm2(x) + ori

        return x


class Transformer(nn.Module):
    def __init__(self, n_layers, embed_dim, head, head_dim, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(nn.ModuleList([
                TransformerBlock(embed_dim=embed_dim, head=head, head_dim=head_dim, mlp_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ViT(nn.Module):
    """
    image_size: int
                Image dimension (the image is supposed to be a square image)
    patch_size: int
                patch dimension
    image_channel: int
                input image channel
    dim: int
                token dim


    """
    def __init__(self, image_size, patch_size, image_channel, pool, dim, head=8, head_dim=64, mlp_dim=64, n_layers=2, dropout=0.1):
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image dimension must be divisible by patch dimension'

        num_patch = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = image_channel * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)  # para:patch_dim*dim
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch+1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(n_layers, dim, head, head_dim, mlp_dim, dropout)

    def forward(self, img):
        # patch embedding: b*c*H*W -> b*(h*w)*(p1*p2*c) -> b*c'*dim
        x = self.to_patch_embedding(img)  # patch token
        b, n, _ = x.shape  # n is number of patch

        # only one cls token, repeat for batch
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)

        # all tokens
        x = torch.cat((cls_tokens, x), dim=1)  # b*(n+1)*dim

        # positional embedding
        x += self.pos_embedding

        # dropout
        x = self.dropout(x)

        # transformer layers
        x = self.transformer(x)
        print(x.shape)

        return x


if __name__ == "__main__":
    img = torch.zeros(1, 3, 256, 256)
    vit = ViT(image_size=256, patch_size=16, image_channel=3, pool='cls', dim=1024)
    out = vit(img)

    transblock = TransformerBlock(embed_dim=512)
    out = transblock(out)

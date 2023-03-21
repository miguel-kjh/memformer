import torch
from einops import rearrange
from torch import nn, einsum

from memformer.Layers.utils import exists, default, max_neg_value


class Attention(nn.Module):
    def __init__(self, dim, heads=8, causal=False, rel_pos_emb=False):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context=None, pos_emb=None, mask=None, query_mask=None, kv_mask=None, attend_self=False):
        b, n, _, h, scale, device = *x.shape, self.heads, self.scale, x.device

        if attend_self:
            kv_input = torch.cat((x, context), dim=1)
        else:
            kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(pos_emb):
            pos_emb_bias = pos_emb(*dots.shape[-2:])
            dots += pos_emb_bias

        mask_value = max_neg_value(dots)

        if self.causal:
            causal_mask = torch.ones((n, n), device=device).triu_(1).bool()
            dots.masked_fill_(causal_mask, mask_value)
            del causal_mask

        if any(map(exists, (query_mask, kv_mask))):
            query_mask = default(query_mask, lambda: torch.ones((b, n), device=device).bool())

            if exists(context):
                kv_mask = default(kv_mask, lambda: torch.ones((b, context.shape[1]), device=device).bool())
            else:
                kv_mask = default(kv_mask, query_mask)

            query_mask = rearrange(query_mask, 'b i -> b () i ()')
            kv_mask = rearrange(kv_mask, 'b j -> b () () j')
            seq_mask = query_mask * kv_mask
            dots.masked_fill_(~seq_mask, mask_value)
            del seq_mask

        if exists(mask):
            mask = rearrange(mask, 'b i j -> b () i j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

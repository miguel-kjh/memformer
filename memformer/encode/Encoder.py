from torch import nn

from memformer.Layers.Attention import Attention
from memformer.Layers.FeedForward import FeedForward
from memformer.Layers.PreNorm import PreNorm
from memformer.Layers.RelativePositionBias import RelativePositionBias
from memformer.Layers.Residual import Residual


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads=8):
        super().__init__()
        self.rel_pos_emb = RelativePositionBias(heads=heads)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, rel_pos_emb=True))),
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))

    def forward(self, x, context=None, src_mask=None):
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x, pos_emb=self.rel_pos_emb, query_mask=src_mask)
            x = cross_attn(x, context=context)
            x = ff(x)
        return x

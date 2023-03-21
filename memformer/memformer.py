import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from collections import namedtuple
from memformer.decode.autoregressive_wrapper import AutoregressiveWrapper
from memformer.Layers.Attention import Attention
from memformer.decode.Decoder import Decoder
from memformer.encode.Encoder import Encoder
from memformer.Layers.FeedForward import FeedForward
from memformer.Layers.PreNorm import PreNorm
from memformer.Layers.Residual import Residual
from memformer.Layers.TransformerWrapper import TransformerWrapper
from memformer.Layers.utils import group_by_key_prefix_and_trim, pick_and_pop, default, exists

# constants

Results = namedtuple('Results', ['enc_out', 'mem', 'dec_out'])
EncOnlyResults = namedtuple('EncOnlyResults', ['enc_out', 'mem'])


class Memformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_memory_slots,
            num_mem_updates=1,
            encoder_only=False,
            mem_update_attn_heads=8,
            **kwargs):
        super().__init__()
        enc_kwargs, kwargs = group_by_key_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = group_by_key_prefix_and_trim('dec_', kwargs)
        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)

        self.encoder = TransformerWrapper(
            dim=dim,
            layer_blocks=Encoder(dim=dim, **enc_kwargs),
            return_logits=False,
            **enc_transformer_kwargs
        )

        self.decoder = TransformerWrapper(
            dim=dim,
            layer_blocks=Decoder(dim=dim, **dec_kwargs),
            return_logits=True,
            **dec_transformer_kwargs
        ) if not encoder_only else None

        if exists(self.decoder):
            self.decoder = AutoregressiveWrapper(self.decoder)

        self.num_mem = num_memory_slots
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, dim))

        self.num_mem_updates = num_mem_updates
        self.mem_updater = Attention(dim, heads=mem_update_attn_heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mem_ff = Residual(PreNorm(dim, FeedForward(dim)))

    def get_initial_mem(self, batch_size):
        return repeat(self.memory_slots, 'n d -> b n d', b=batch_size)

    def forward(self, src, tgt=None, mems=None, src_mask=None, tgt_mask=None):
        b, n, num_mem, device = *src.shape, self.num_mem, src.device
        mems = default(mems, lambda: self.get_initial_mem(b))

        enc = self.encoder(src, context=mems, src_mask=src_mask)

        if exists(self.decoder) and exists(tgt):
            dec_out = self.decoder(tgt, context=enc, src_mask=tgt_mask, tgt_mask=src_mask, return_loss=True)
        else:
            dec_out = torch.tensor(0., requires_grad=True, device=device)

        # update memory with attention
        mem_mask = torch.eye(num_mem, num_mem, device=device).bool()
        mem_mask = repeat(mem_mask, 'i j -> b i j', b=b)
        mem_mask = F.pad(mem_mask, (0, n), value=True)

        if exists(src_mask):
            src_mask = rearrange(src_mask, 'b j -> b () j')
            mem_enc_mask = F.pad(src_mask, (num_mem, 0), value=True)
            mem_mask &= mem_enc_mask

        for _ in range(self.num_mem_updates):
            prev_mems = mems
            updated_mems = self.mem_updater(mems, enc, mask=mem_mask, attend_self=True)

            next_mems = self.gru(
                rearrange(updated_mems, 'b n d -> (b n) d'),
                rearrange(prev_mems, 'b n d -> (b n) d')
            )

            mems = rearrange(next_mems, '(b n) d -> b n d', b=b)
            mems = self.mem_ff(mems)

        if not exists(self.decoder):
            return EncOnlyResults(enc, mems)

        return Results(enc, mems, dec_out)

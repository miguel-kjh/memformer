from torch import nn


class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, dim, layer_blocks, heads=8, return_logits=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len
        self.layer_blocks = layer_blocks
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else nn.Identity()

    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x = self.layer_blocks(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)

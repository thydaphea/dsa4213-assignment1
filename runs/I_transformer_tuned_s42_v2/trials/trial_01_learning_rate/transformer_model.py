import torch
from torch import nn

class CausalTransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.max_length = config['block_length']
        d = config['embedding_dim']
        if d % config['nhead'] != 0:
            raise ValueError('Embedding dimension must be divisible by the head count.')
        self.embedding = nn.Embedding(config['vocab_size'], d, padding_idx=0)
        self.position = nn.Embedding(self.max_length, d)
        self.input_dropout = nn.Dropout(config['dropout'])
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'], dropout=config['dropout'],
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(config['num_layers'])])
        self.final_norm = nn.LayerNorm(d)
        self.output_dropout = nn.Dropout(config['dropout'])
        self.output = nn.Linear(d, config['vocab_size'])
        forbidden = torch.zeros(config['vocab_size'], dtype=torch.bool)
        forbidden[0] = True; forbidden[2] = True
        self.register_buffer('input_only_mask', forbidden)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        with torch.no_grad(): self.embedding.weight[0].zero_()

    def token_states(self, x):
        if x.ndim != 2 or not 0 < x.size(1) <= self.max_length:
            raise ValueError('Input must be [batch, length] within the saved context limit.')
        if torch.any(x[:, 0] == 0):
            raise ValueError('Use nonempty sequences with right padding; no left padding.')
        t = x.size(1)
        positions = torch.arange(t, device=x.device)
        h = self.input_dropout(self.embedding(x) + self.position(positions)[None, :, :])
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        for block in self.blocks:
            h = block(h, src_mask=causal, src_key_padding_mask=x.eq(0))
        return self.final_norm(h)

    def forward(self, x, return_states=False):
        states = self.token_states(x)
        logits = self.output(self.output_dropout(states))
        logits = logits.masked_fill(self.input_only_mask, float('-inf'))
        return (logits, states) if return_states else logits

    def encode_mean(self, x):
        states = self.token_states(x)
        mask = (x != 0) & (x != 2) & (x != 3)
        if torch.any(mask.sum(1) == 0): raise ValueError('No lexical token to pool.')
        return (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

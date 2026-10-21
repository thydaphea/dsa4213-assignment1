import torch
from torch import nn

class VanillaRNNLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'], padding_idx=0)
        self.input_dropout = nn.Dropout(config['dropout'])
        self.rnn = nn.RNN(config['embedding_dim'], config['hidden_size'],
                            num_layers=config['num_layers'], batch_first=True,
                            bidirectional=False, nonlinearity='tanh', dropout=0.0)
        self.output_dropout = nn.Dropout(config['dropout'])
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        mask = torch.zeros(config['vocab_size'], dtype=torch.bool)
        mask[0] = True; mask[2] = True
        self.register_buffer('input_only_mask', mask)

    def token_states(self, x):
        # No state argument: PyTorch initializes the hidden state to zero each call.
        states, _ = self.rnn(self.input_dropout(self.embedding(x)))
        return states

    def forward(self, x, return_states=False):
        states = self.token_states(x)
        logits = self.output(self.output_dropout(states))
        logits = logits.masked_fill(self.input_only_mask, float('-inf'))
        return (logits, states) if return_states else logits

    def encode_mean(self, x):
        # Use eval() and inference_mode() when extracting frozen representations.
        states = self.token_states(x)
        mask = (x != 0) & (x != 2) & (x != 3)
        if torch.any(mask.sum(1) == 0): raise ValueError('No lexical token to pool.')
        return (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

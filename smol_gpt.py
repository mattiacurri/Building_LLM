import torch
import torch.nn as nn
from SelfAttention import MultiHeadAttention

# Configuration of SmolGPT
SMOLGPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context size
    "embed_dim": 768, # Embedding size
    "n_layers": 12, # Number of layers
    "n_heads": 12, # Number of attention heads
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False, # Query-Key-Value bias
}

class SmolGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformers_layers = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln = LayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, sequence_length = x.size()
        x = self.token_embedding(x) + self.position_embedding(torch.arange(sequence_length, device=x.device))
        x = self.dropout(x)
        x = self.transformers_layers(x)
        x = self.ln(x)
        logits = self.out_head(x)
        return logits


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        '''
        - Scale: allows the model to adjust the variance of the normalized output. 
        Without this, the output would always have unit variance, which might not be optimal for all layers.
        - Shift: allows the model to adjust the mean of the normalized output. 
        Without this, the output would always have zero mean, which might not be optimal for all layers.
        
        By learning these parameters, the model can retain the flexibility to represent the original input distribution, while still benefiting from the normalization process.
        '''
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False means to not use the Bessel's correction, which tipically uses n-1 instead of n in the denominator of variance formula to adjust for bias
        # given that the embedding size is significantly large, the difference between n and n-1 is negligible
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU(x) = x * phi(x), where phi(x) is the standard Gaussian cumulative distribution function
        # For a faster approximation, we can use the following formula
        return 0.5 * x * (1 + torch.tanh((torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # (batch_size, num_tokens, embed_dim) -> (batch_size, num_tokens, 4 * embed_dim) -> (batch_size, num_tokens, embed_dim)
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]), # input: (2, 3, 768) -> output: (2, 3, 3072) 
            GELU(), # input: (2, 3, 3072) -> output: (2, 3, 3072)
            nn.Linear(cfg["embed_dim"] * 4, cfg["embed_dim"]) # input: (2, 3, 3072) -> output: (2, 3, 768)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_in=cfg["embed_dim"], 
            d_out=cfg["embed_dim"], 
            context_length=cfg["context_length"], 
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"], 
            qkv_bias=cfg["qkv_bias"])
        self.feed_forward = FeedForward(cfg)
        self.layer_norm_1 = LayerNorm(cfg["embed_dim"])
        self.layer_norm_2 = LayerNorm(cfg["embed_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Block 1
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x += shortcut # Residual connection

        # Block 2
        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x += shortcut # Residual connection
        return x

# Test of LayerNorm
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
ln = LayerNorm(embed_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(-1, keepdim=True)
var = out_ln.var(-1, keepdim=True, unbiased=False)
print(f'Mean: {mean}')
print(f'Variance: {var}')

# Test of TransformerBlock
torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = TransformerBlock(SMOLGPT_CONFIG_124M)
out_block = block(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {out_block.shape}')

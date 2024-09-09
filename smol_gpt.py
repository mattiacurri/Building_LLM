import torch
import torch.nn as nn

# Configuration of SmolGPT
SMOLGPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context size
    "embedding_size": 768, # Embedding size
    "n_layers": 12, # Number of layers
    "n_heads": 12, # Number of attention heads
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False, # Query-Key-Value bias
}

class SmolGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_size"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_size"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformers_layers = nn.Sequential(*[DummyTransformerLayer(cfg) for _ in range(cfg["n_layers"])])
        self.ln = LayerNorm(cfg["embedding_size"])
        self.out_head = nn.Linear(cfg["embedding_size"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, sequence_length = x.size()
        x = self.token_embedding(x) + self.position_embedding(torch.arange(sequence_length, device=x.device))
        x = self.dropout(x)
        x = self.transformers_layers(x)
        x = self.ln(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.eps = 1e-5
        '''
        - Scale: allows the model to adjust the variance of the normalized output. 
        Without this, the output would always have unit variance, which might not be optimal for all layers.
        - Shift: allows the model to adjust the mean of the normalized output. 
        Without this, the output would always have zero mean, which might not be optimal for all layers.
        
        By learning these parameters, the model can retain the flexibility to represent the original input distribution, while still benefiting from the normalization process.
        '''
        self.scale = nn.Parameter(torch.ones(embedding_size))
        self.shift = nn.Parameter(torch.zeros(embedding_size))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False means to not use the Bessel's correction, which tipically uses n-1 instead of n in the denominator of variance formula to adjust for bias
        # given that the embedding size is significantly large, the difference between n and n-1 is negligible
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# Test of LayerNorm
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
ln = LayerNorm(embedding_size=5)
out_ln = ln(batch_example)
mean = out_ln.mean(-1, keepdim=True)
var = out_ln.var(-1, keepdim=True, unbiased=False)
print(f'Mean: {mean}')
print(f'Variance: {var}')

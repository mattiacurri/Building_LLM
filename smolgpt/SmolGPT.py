import torch
import torch.nn as nn

import sys
import os

# add in sys.path the path of the folder containing the module
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_attention.SelfAttention import MultiHeadAttention

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
        # Final linear layer to come back to vocabulary size in order to produce logits
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

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# Now let's use some decoding strategies to generate better text (temperature scaling and top-k sampling)
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # top-k sampling: maintain only the top k highest probabilities
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_val = top_logits[:, -1]
            # mask with -inf to exclude the values below the top-k
            logits = torch.where(logits < min_val, torch.tensor(-float('inf')).to(logits.device), logits)
        # temperature scaling: increase the temperature to make the distribution more uniform or decrease it to make it more peaky (more confident)
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        if next_token == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        idx = torch.cat((idx, next_token), dim=1)
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
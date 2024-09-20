import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
    
    def forward(self, x):
        queries = torch.matmul(x, self.W_query)
        keys = torch.matmul(x, self.W_key)
        values = torch.matmul(x, self.W_value)
        attention_scores = torch.matmul(queries, keys.T)
        attention_weights = torch.softmax(attention_scores/self.d_out**0.5, dim=1)
        context_vectors = torch.matmul(attention_weights, values)
        return context_vectors

# Improvement: SelfAttentionV2
# Instead of nn.Parameter, we can use nn.Linear to define the query, key, and value weight matrices
# nn.Linear use an optimized weight initialization scheme, which can lead to faster convergence during training and stability
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attention_scores = torch.matmul(queries, keys.T)
        attention_weights = torch.softmax(attention_scores/self.d_out**0.5, dim=1)
        context_vectors = torch.matmul(attention_weights, values)
        return context_vectors

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, context_length, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # buffers are automatically moved to the appropriate device along with our model
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) 

    def forward(self, x):
        # b: batch dimension
        # num_tokens: number of tokens in the sequence
        # d_in: input dimension
        # handle batch size = 1
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        b, num_tokens, d_in = x.shape  

        queries = self.W_query(x)  # Project x to query space
        keys = self.W_key(x)       # Project x to key space
        values = self.W_value(x)   # Project x to value space

        # Attention Scores
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))  # keys.T(1, 2): trasposta di keys rispetto alle dimensioni 1 e 2

        # Causal Attention
        # self.mask.bool()[:num_tokens, :num_tokens] output --> tensor([[False,  True,  True,  True,  True,  True], ...]) of size (num_tokens, num_tokens)
        attention_scores = attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))

        # Softmax to obtain attention weights
        attention_weights = torch.softmax(attention_scores / self.d_out**0.5, dim=1)

        # Dropout
        attention_weights = self.dropout(attention_weights)

        # Context vectors
        context_vectors = torch.matmul(attention_weights, values)

        return context_vectors

# Sequential Multi-Head Attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, dropout, context_length, qkv_bias) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# Parallel Multi-Head Attention - Split the input into multiple heads and process them in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "Number of heads must divide output dimension"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to project the concatenated heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # (b, num_heads, num_tokens, head_dim) x (b, num_heads, head_dim, num_tokens) -> (b, num_heads, num_tokens, num_tokens)
        attn_scores = torch.matmul(queries, keys.transpose(2, 3))  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection, standard convention in LLM implementation, but it's not strictly necessary (recent research has shown that it can be removed without affecting the modeling performance)

        return context_vec
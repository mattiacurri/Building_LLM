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

# Test
torch.manual_seed(123)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1) 
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
)
d_in = inputs.shape[1]
d_out = 2
sa = SelfAttentionV1(3, 2)
print(sa(inputs))

sa2 = SelfAttentionV2(3, 2)
print(sa2(inputs))
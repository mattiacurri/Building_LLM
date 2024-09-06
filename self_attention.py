# Chapter 3 - Self-Attention Mechanisms
import torch
import timeit


# 3.3.1 Simplified Self-Attention without trainable weights

# Your journey starts with one step is our input sequence
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1) 
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
)

# First step: Calculate the dot product of the query (x^2 as example) with every other input token
query = inputs[1] # journey
attention_scores_2 = torch.matmul(inputs, query) # for i, x_i in enumerate(inputs): attention_scores_2[i] = torch.dot(x_i, query) 
print(f'Attention Scores for "journey": {attention_scores_2}')

# Second step: Normalize the attention scores
attention_scores_2 = torch.softmax(attention_scores_2, dim=0) # apply softmax to get the normalized attention scores
print(f'Normalized Attention Scores for "journey": {attention_scores_2}')

# Third Step: Compute the context vector
context_vector_2 = torch.zeros(query.shape) # initialize the context vector
for i, x_i in enumerate(inputs):
    context_vector_2 += attention_scores_2[i] * x_i # weighted sum of the input tokens, where the weights are the normalized attention scores
print(f'Context Vector for "journey": {context_vector_2}')

# Now we generalize the above steps to compute the context vectors for all input tokens
attention_scores = [] * len(inputs)
context_vectors = [] * len(inputs)

# Sample input tensor
inputs_ = torch.randn(100, 64)  # Adjust the size as needed

def my_implementation(inputs):
    attention_scores = []
    context_vectors = []
    for q in inputs:
        a_q = torch.matmul(inputs, q)
        a_q = torch.softmax(a_q, dim=0)
        attention_scores.append(a_q)  # store the attention scores for the query q
        z_q = torch.zeros(q.shape)
        for i, x_i in enumerate(inputs):
            z_q += a_q[i] * x_i
        context_vectors.append(z_q)  # store the context vector for the query q
    return attention_scores, context_vectors

def book_implementation(inputs):
    attn_scores = torch.matmul(inputs, inputs.T)  # compute the dot product of the input tensor with every other input token
    attn_weights = torch.softmax(attn_scores, dim=1)  # apply softmax to normalize the attention scores
    all_context_vectors = torch.matmul(attn_weights, inputs)  # compute the context vectors
    return attn_weights, all_context_vectors

# Benchmark the original implementation
original_time = timeit.timeit(lambda: my_implementation(inputs_), number=4)

# Benchmark the book implementation
book_time = timeit.timeit(lambda: book_implementation(inputs_), number=4)

print(f"My implementation: {original_time:.6f} seconds")
print(f"Book implementation: {book_time:.6f} seconds") # obviously the book implementation is faster 'cause it uses matrix multiplication instead of for loops

# 3.4 Self-Attention with Trainable Weights
torch.manual_seed(123)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 # in GPT-like models d_in = d_out, but for semplicity we set d_out = 2

# Note: requires_grad = False to reduce clutter in the outputs, but in practice, these weights would be trainable (requires_grad = True)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # trainable query weight matrix
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # trainable key weight matrix
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # trainable value weight matrix

query_2 = torch.matmul(x_2, W_query) # project the query x^2 into the query space
key_2 = torch.matmul(x_2, W_key) # project all input tokens into the key space
value_2 = torch.matmul(x_2, W_value) # project all input tokens into the value space
print(query_2)

# First step: compute keys and values for all input tokens
keys = torch.matmul(inputs, W_key) # project all input tokens into the key space
values = torch.matmul(inputs, W_value) # project all input tokens into the value space
print(f"Keys shape: {keys.shape}")
print(f"Values shape: {values.shape}")

# Let's compute the attention score w22 for the query x^2 and key x^2
keys_2 = keys[1] # key for x^2
# instead of using directly the input token x^2 we use the projected key
attention_score_22 = torch.dot(query_2, keys_2) # compute the dot product of the query and key
print(f'Attention Score for "journey": {attention_score_22:.4f}')

# Second step: compute the attention scores for the query x^2 with all input tokens
attention_scores_2 = torch.matmul(query_2, keys.T) # compute the dot product of the query with every key
print(f'Attention Scores for "journey": {attention_scores_2}')

# Third step: going from attention scres to attention weights --> softmax
d_k = keys.shape[1] # get the dimension of the key space
attention_scores_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=0) # apply softmax to get the normalized attention scores
# the difference to earlier is that we now scale the attention scores by dividing by the square root of the dimension of the key space
# this is a common practice in self-attention mechanisms to stabilize the gradients, as the dot product grows with the dimensionality of the key space
print(f'Normalized Attention Scores for "journey": {attention_scores_2}')

# Fourth step: compute the context vector
# To do so we multiply each value vector with its respective attention weight and then summing them to obtain the context vector
context_vector_2 = torch.matmul(attention_scores_2, values)
print(context_vector_2)

'''
The terms "key", "value", and "query" are used to draw an analogy to information retrieval systems:

1. **Query**: represents the vector for which we are trying to find relevant information. 

2. **Key**: represents the vector that is used to match against the query.
The similarity between the query and the key determines the attention score.

3. **Value**: represents the actual information that is being retrieved.
The attention scores are used to weight these values to produce the final output.
'''
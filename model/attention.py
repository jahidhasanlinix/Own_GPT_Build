import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    """
    Single-head self-attention with causal masking for autoregressive modeling.
    
    This module implements the core attention mechanism from the Transformer
    architecture, designed for causal language modeling where each position
    can only attend to previous positions (not future ones).
    
    Key Features:
    - Self-attention (queries, keys, values all from same input)
    - Causal masking (prevents looking at future tokens)
    - Scaled dot-product attention
    - Learnable linear projections for Q, K, V
    
    Mathematical Foundation:
    -----------------------
    1. Project inputs to Q, K, V: Q = X·W_q, K = X·W_k, V = X·W_v
    2. Compute attention scores: S = Q·K^T / √d_k
    3. Apply causal mask (upper triangle = -∞)
    4. Softmax normalization: A = softmax(S, dim=-1)
    5. Weighted sum: Output = A·V
    
    Where:
    - X: input embeddings (batch_size, seq_len, embedding_dim)
    - W_q, W_k, W_v: learnable weight matrices (embedding_dim, attention_dim)
    - d_k: attention_dim (scaling factor)
    """
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        """
        Initializes the single-head attention module.
        
        Parameters
        ----------
        embedding_dim : int
            Dimension of input token embeddings (d_model in transformer papers).
            Typical values: 256, 512, 768, 1024
            
        attention_dim : int
            Dimension of query, key, and value projections (d_k in attention).
            Often set to embedding_dim // num_heads for multi-head attention.
            Controls the representational capacity of attention.
            
        Layer Details:
        -------------
        key_gen : nn.Linear(embedding_dim, attention_dim, bias=False)
            Projects input embeddings to keys (K)
            No bias for cleaner attention computation
            
        query_gen : nn.Linear(embedding_dim, attention_dim, bias=False)
            Projects input embeddings to queries (Q)
            No bias for cleaner attention computation
            
        value_gen : nn.Linear(embedding_dim, attention_dim, bias=False)
            Projects input embeddings to values (V)
            No bias for cleaner attention computation
            
        Notes
        -----
        - Fixed random seed (0) for reproducible initialization
        - No biases in projections (standard for attention in some implementations)
        - This is single-head attention (not multi-head)
        """
        super().__init__()
        
        # Set random seed for reproducible weight initialization
        torch.manual_seed(0)
        
        # Key projection: embedding_dim → attention_dim
        # Maps each token to a "key" used for matching with queries
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        
        # Query projection: embedding_dim → attention_dim
        # Maps each token to a "query" used to attend to keys
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        
        # Value projection: embedding_dim → attention_dim
        # Maps each token to "values" that are aggregated based on attention
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """
        Performs causal self-attention on the input sequence.
        
        The forward pass computes attention where each position can only
        attend to itself and previous positions (causal/autoregressive masking).
        
        Parameters
        ----------
        embedded : TensorType[float]
            Input token embeddings, shape (batch_size, context_length, embedding_dim).
            - batch_size: Number of sequences in batch
            - context_length: Length of each sequence (L)
            - embedding_dim: Size of embedding vectors (d_model)
            
        Returns
        -------
        TensorType[float]
            Attention output, shape (batch_size, context_length, attention_dim),
            rounded to 4 decimal places.
            Each position contains a weighted sum of values from previous positions.
            
        Mathematical Steps:
        ------------------
        1. Q = X·W_q, K = X·W_k, V = X·W_v  [shape: (B, L, A)]
        2. scores = Q·K^T / sqrt(A)         [shape: (B, L, L)]
        3. Apply causal mask (upper triangular = -∞)
        4. attention_weights = softmax(scores, dim=-1)  [shape: (B, L, L)]
        5. output = attention_weights · V   [shape: (B, L, A)]
        
        Examples
        --------
        >>> attention = SingleHeadAttention(embedding_dim=512, attention_dim=64)
        >>> attention.eval()
        
        # Single sequence (batch_size=1, seq_len=10, emb_dim=512)
        >>> x = torch.randn(1, 10, 512)
        >>> output = attention(x)
        >>> output.shape
        torch.Size([1, 10, 64])
        
        # Batch processing
        >>> batch = torch.randn(32, 20, 512)  # 32 sequences, length 20
        >>> outputs = attention(batch)
        >>> outputs.shape
        torch.Size([32, 20, 64])
        
        # Attention pattern shows causal structure
        >>> context_len = 5
        >>> x = torch.randn(1, context_len, 512)
        >>> output = attention(x)
        >>> # Position 4 attends to positions 0,1,2,3,4
        >>> # Position 0 attends only to position 0
        """
        
        # ========== STEP 1: Generate Queries, Keys, Values ==========
        # Project input embeddings to query space
        # Shape: (batch_size, context_length, attention_dim)
        q = self.query_gen(embedded)
        
        # Project input embeddings to key space
        # Shape: (batch_size, context_length, attention_dim)
        k = self.key_gen(embedded)
        
        # Project input embeddings to value space
        # Shape: (batch_size, context_length, attention_dim)
        v = self.value_gen(embedded)
        
        # ========== STEP 2: Compute Attention Scores ==========
        # Matrix multiplication: Q @ K^T
        # For each query position, compute dot product with all key positions
        # This gives similarity scores between each pair of positions
        # scores[b, i, j] = Q[b, i] · K[b, j]
        scores = q @ torch.transpose(k, 1, 2)
        # Shape: (batch_size, context_length, context_length)
        # Higher scores = more attention between positions
        
        # ========== STEP 3: Scale the Scores ==========
        # Scaling by 1/√d_k prevents dot products from growing too large
        # For large d_k, dot products can become large, pushing softmax
        # into regions of very small gradients (vanishing gradient problem)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scaling_factor = attention_dim ** 0.5  # sqrt(d_k)
        scores = scores / scaling_factor
        
        # ========== STEP 4: Apply Causal Mask ==========
        # Create lower triangular matrix (including diagonal)
        # shape: (context_length, context_length)
        # Example for context_length=4:
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        
        # Create mask where positions are 0 (should be masked out)
        # mask[i, j] = True if position j > i (future token)
        mask = lower_triangular == 0
        
        # Fill masked positions with -inf
        # After softmax, exp(-inf) = 0, so future positions get zero attention
        scores = scores.masked_fill(mask, float('-inf'))
        
        # ========== STEP 5: Apply Softmax Normalization ==========
        # Convert scores to probabilities (attention weights)
        # dim=2 means normalize across key positions (columns)
        # For each query position, weights sum to 1
        attention_weights = nn.functional.softmax(scores, dim=2)
        # Shape: (batch_size, context_length, context_length)
        # attention_weights[b, i, j] = attention from position i to j
        
        # ========== STEP 6: Compute Weighted Sum of Values ==========
        # Output = attention_weights @ V
        # Each output position is a weighted combination of all value vectors
        # (but causal mask ensures only current/previous positions contribute)
        output = attention_weights @ v
        # Shape: (batch_size, context_length, attention_dim)
        
        # Round to 4 decimal places for consistent output
        return torch.round(output, decimals=4)

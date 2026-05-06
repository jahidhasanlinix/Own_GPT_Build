import torch
from typing import List, Tuple

class Solution:
    """
    Batch data loader for training language models with next-token prediction.
    
    This class creates random batches of sequence pairs (X, Y) where Y is
    the target sequence shifted by one position relative to X. This setup
    allows the model to learn to predict the next token at each position.
    
    Key Features:
    - Random sampling of starting positions (not sequential)
    - Variable-length sequences padded to same length
    - Fixed random seed for reproducibility
    - Token-level processing (simple whitespace tokenization)
    
    Training Task:
    -------------
    Given input sequence X = [x₁, x₂, ..., xₙ],
    predict target sequence Y = [x₂, x₃, ..., xₙ₊₁]
    
    This is equivalent to learning the conditional probability:
    P(x_{t+1} | x₁, x₂, ..., x_t)
    
    Common Use Cases:
    ----------------
    - Training GPT-style language models
    - Next-word prediction tasks
    - Autoregressive sequence modeling
    - Character-level text generation
    """
    
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        """
        Generate a random batch of input-target sequence pairs.
        
        The function creates training examples by:
        1. Tokenizing the raw text into words
        2. Randomly selecting batch_size starting positions
        3. Extracting context_length consecutive tokens as input (X)
        4. Extracting the next context_length tokens as target (Y)
        
        Parameters
        ----------
        raw_dataset : str
            Raw text corpus to create sequences from.
            The entire text is used as one continuous stream.
            Example: "The quick brown fox jumps over the lazy dog"
            
        context_length : int
            Length of each input/target sequence (number of tokens).
            Also known as "sequence length" or "window size".
            Typical values: 32, 64, 128, 256, 512
            Must be less than total number of tokens.
            
        batch_size : int
            Number of sequence pairs to generate per batch.
            Controls how many independent starting positions are sampled.
            Typical values: 16, 32, 64, 128
            
        Returns
        -------
        Tuple[List[List[str]], List[List[str]]]
            A tuple containing:
            - X: List of input sequences (batch_size × context_length)
            - Y: List of target sequences (batch_size × context_length)
            Each sequence is a list of string tokens.
            
        Mathematical Details:
        --------------------
        Let T = tokenized text of length L.
        For each sampled index i in range(0, L - context_length):
            X_i = [T[i], T[i+1], ..., T[i+context_length-1]]
            Y_i = [T[i+1], T[i+2], ..., T[i+context_length]]
        
        The model learns to minimize:
            Loss = Σₜ CrossEntropy(Y_i[t], model(X_i)[t])
            
        Examples
        --------
        >>> loader = Solution()
        >>> text = "a b c d e f g h i j"
        
        >>> # Create a batch of 2 sequences, each of length 3
        >>> X, Y = loader.batch_loader(text, context_length=3, batch_size=2)
        
        >>> print("Input sequences (X):")
        >>> for seq in X:
        ...     print(seq)
        ['b', 'c', 'd']  # Example output (indices may vary)
        ['e', 'f', 'g']
        
        >>> print("Target sequences (Y):")
        >>> for seq in Y:
        ...     print(seq)
        ['c', 'd', 'e']
        ['f', 'g', 'h']
        
        Notes
        -----
        - Uses random sampling (not sequential batching)
        - Fixed random seed (0) ensures reproducibility
        - Simple whitespace tokenization (splits on spaces)
        - No padding needed as all sequences have same length
        - Each batch contains independent, randomly sampled sequences
        """
        
        # ========== STEP 1: SET RANDOM SEED ==========
        # Ensures reproducible results across runs
        # Without this, each run would generate different random indices
        torch.manual_seed(0)
        
        # ========== STEP 2: TOKENIZATION ==========
        # Split raw text into tokens (words) by whitespace
        # This is a simple tokenizer - real applications use more sophisticated ones
        # Example: "Hello world!" → ["Hello", "world!"]
        tokenized = raw_dataset.split()
        
        # Calculate total number of tokens
        total_tokens = len(tokenized)
        
        # ========== STEP 3: VALIDATE INPUTS ==========
        # Ensure we have enough tokens to extract sequences
        if total_tokens <= context_length:
            raise ValueError(
                f"Text too short: {total_tokens} tokens, need at least {context_length + 1}"
            )
        
        # ========== STEP 4: GENERATE RANDOM STARTING POSITIONS ==========
        # Valid starting indices range from 0 to total_tokens - context_length - 1
        # We need at least context_length+1 tokens for X and Y
        max_start = total_tokens - context_length - 1
        
        # Randomly sample batch_size indices without replacement
        # size=(batch_size,) creates 1D tensor of shape (batch_size,)
        indices = torch.randint(
            low=0,                    # Minimum index (inclusive)
            high=total_tokens - context_length,  # Maximum index (exclusive)
            size=(batch_size,),       # Number of indices to generate
            generator=None           # Use default RNG (already seeded)
        ).tolist()  # Convert from tensor to Python list for iteration
        
        # ========== STEP 5: CREATE INPUT AND TARGET SEQUENCES ==========
        X = []  # Input sequences
        Y = []  # Target sequences (shifted by 1)
        
        for idx in indices:
            # Extract input sequence: tokens[idx : idx+context_length]
            # This gives us context_length consecutive tokens
            input_seq = tokenized[idx:idx + context_length]
            X.append(input_seq)
            
            # Extract target sequence: tokens[idx+1 : idx+1+context_length]
            # This is the input sequence shifted right by one position
            # The model learns to predict the next token at each position
            target_seq = tokenized[idx+1:idx+1 + context_length]
            Y.append(target_seq)
        
        return X, Y

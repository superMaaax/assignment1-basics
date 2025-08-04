from __future__ import annotations
from jaxtyping import Float, Int
from torch import Tensor
import torch
import math
import torch.nn.functional as F

def linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    return in_features @ weights.T

def embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    return weights[token_ids]

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    
    # SwiGLU formula: SiLU(xW1) ⊙ (xW3) projected by W2
    
    # First linear transformation with SiLU activation
    w1_out = in_features @ w1_weight.T  # (..., d_ff)
    silu_w1 = silu(w1_out)  # SiLU activation on W1
    
    # Second linear transformation (gate)
    w3_out = in_features @ w3_weight.T  # (..., d_ff)
    
    # Element-wise multiplication (gating mechanism)
    gated = silu_w1 * w3_out  # (..., d_ff)
    
    # Final linear transformation (down-projection)
    output = gated @ w2_weight.T  # (..., d_model)

    return output


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
    is_causal: bool = False,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Optimized scaled dot product attention implementation with causal masking support.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
        is_causal (bool): Whether to apply causal (autoregressive) masking
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    
    # Get dimensions
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    scale = 1.0 / math.sqrt(d_k)
    
    # Compute scaled scores
    scores = Q @ K.transpose(-2, -1)
    scores = scores * scale
    
    # Apply causal mask if requested
    if is_causal:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        causal_mask_value = torch.where(causal_mask, 0.0, float('-inf'))
        scores = scores + causal_mask_value
    # Apply additional mask if provided
    if mask is not None:
        mask_value = torch.where(mask == 0, float('-inf'), 0.0)
        scores = scores + mask_value
    
    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Final output
    output = attention_weights @ V
    
    return output

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    *batch_size, sequence_length, d_in = in_features.shape
    d_head = d_model // num_heads
    Q = in_features @ q_proj_weight.T  # (..., seq_length, d_k)
    K = in_features @ k_proj_weight.T  # (..., seq_length, d_k)
    V = in_features @ v_proj_weight.T  # (..., seq_length, d_v)

    Q = Q.view(*batch_size, sequence_length, num_heads, d_head).transpose(-3, -2)  # (..., num_heads, seq_length, d_head)
    K = K.view(*batch_size, sequence_length, num_heads, d_head).transpose(-3, -2)  # (..., num_heads, seq_length, d_head)
    V = V.view(*batch_size, sequence_length, num_heads, d_head).transpose(-3, -2)  # (..., num_heads, seq_length, d_head)

    attn_output = scaled_dot_product_attention(Q, K, V, is_causal=True)  # (..., num_heads, seq_length, d_head)
    attn_output = attn_output.transpose(-3, -2).contiguous()  # (..., seq_length, num_heads, d_head)
    attn_output = attn_output.view(*batch_size, sequence_length, d_model)  # (..., seq_length, d_model)
    attn_output = attn_output @ o_proj_weight.T  # (..., seq_length, d_model)
    return attn_output


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # Get device and shape info
    device = in_query_or_key.device
    *batch_dims, seq_len, _ = in_query_or_key.shape
    
    # Create frequency indices: [0, 1, 2, ..., d_k/2-1]
    freq_indices = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
    
    # Compute frequencies: theta^(-2i/d_k) for i in [0, 1, ..., d_k/2-1]
    freqs = theta ** (-2.0 * freq_indices / d_k)
    
    # Expand token_positions to match batch dimensions if needed
    if token_positions.dim() == 1:
        # If token_positions is 1D, expand it to match batch dimensions
        for _ in batch_dims:
            token_positions = token_positions.unsqueeze(0)
    
    # Compute position-frequency matrix: outer product of positions and frequencies
    # token_positions: (..., seq_len), freqs: (d_k//2) -> (..., seq_len, d_k//2)
    pos_freqs = token_positions.unsqueeze(-1).float() * freqs.unsqueeze(0)
    
    # Compute cos and sin
    cos_pos = torch.cos(pos_freqs)  # (..., seq_len, d_k//2)
    sin_pos = torch.sin(pos_freqs)  # (..., seq_len, d_k//2)
    
    # Split input into even and odd indices
    x_even = in_query_or_key[..., 0::2]  # (..., seq_len, d_k//2)
    x_odd = in_query_or_key[..., 1::2]   # (..., seq_len, d_k//2)
    
    # Apply rotation: 
    # x_even_rot = x_even * cos - x_odd * sin
    # x_odd_rot = x_even * sin + x_odd * cos
    x_even_rot = x_even * cos_pos - x_odd * sin_pos
    x_odd_rot = x_even * sin_pos + x_odd * cos_pos
    
    # Interleave back: [x_even_rot[0], x_odd_rot[0], x_even_rot[1], x_odd_rot[1], ...]
    output = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (..., seq_len, d_k//2, 2)
    output = output.flatten(-2)  # (..., seq_len, d_k)
    
    return output


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Get dimensions
    *batch_dims, seq_len, d_in = in_features.shape
    d_head = d_model // num_heads
    
    # Generate default token positions if not provided
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)
        # Expand to match batch dimensions
        for _ in batch_dims:
            token_positions = token_positions.unsqueeze(0)
    
    # Project to Q, K, V
    Q = in_features @ q_proj_weight.T  # (..., seq_len, d_model)
    K = in_features @ k_proj_weight.T  # (..., seq_len, d_model)
    V = in_features @ v_proj_weight.T  # (..., seq_len, d_model)
    
    # Reshape for multi-head attention
    Q = Q.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
    K = K.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
    V = V.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
    
    # Apply RoPE to Q and K (but not V)
    Q = rope(d_head, theta, max_seq_len, Q, token_positions)
    K = rope(d_head, theta, max_seq_len, K, token_positions)
    
    # Apply scaled dot product attention with causal masking
    attn_output = scaled_dot_product_attention(Q, K, V, is_causal=True)
    
    # Reshape back and apply output projection
    attn_output = attn_output.transpose(-3, -2).contiguous()  # (..., seq_len, num_heads, d_head)
    attn_output = attn_output.view(*batch_dims, seq_len, d_model)  # (..., seq_len, d_model)
    attn_output = attn_output @ o_proj_weight.T  # (..., seq_len, d_model)
    
    return attn_output


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    
    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]): State dict with transformer block weights
        in_features (Float[Tensor, "batch sequence_length d_model"]): Input tensor

    Returns:
        Float[Tensor, "batch sequence_length d_model"]: Output tensor
    """
    # Extract weights
    q_proj_weight = weights["attn.q_proj.weight"]
    k_proj_weight = weights["attn.k_proj.weight"] 
    v_proj_weight = weights["attn.v_proj.weight"]
    o_proj_weight = weights["attn.output_proj.weight"]
    ln1_weight = weights["ln1.weight"]
    w1_weight = weights["ffn.w1.weight"]
    w2_weight = weights["ffn.w2.weight"]
    w3_weight = weights["ffn.w3.weight"]
    ln2_weight = weights["ln2.weight"]
    
    # Pre-norm architecture:
    # x = x + self_attention(layer_norm1(x))
    # x = x + feedforward(layer_norm2(x))
    
    # Self-attention block with residual connection
    normed1 = rmsnorm(d_model, 1e-5, ln1_weight, in_features)
    attn_output = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=normed1,
        token_positions=None  # Use default positions
    )
    x = in_features + attn_output  # Residual connection
    
    # Feed-forward block with residual connection
    normed2 = rmsnorm(d_model, 1e-5, ln2_weight, x)
    ffn_output = swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=normed2
    )
    x = x + ffn_output  # Residual connection
    
    return x


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE Θ parameter.
        weights (dict[str, Tensor]): State dict of the transformer model
        in_indices (Int[Tensor, "batch_size sequence_length"]): Input token indices

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Output logits
    """
    # Token embedding
    token_embedding_weight = weights["token_embeddings.weight"]
    x = embedding(vocab_size, d_model, token_embedding_weight, in_indices)
    
    # Apply transformer blocks
    for layer_idx in range(num_layers):
        # Extract weights for this layer
        layer_weights = {}
        for key, value in weights.items():
            if key.startswith(f"layers.{layer_idx}."):
                # Remove the layer prefix
                new_key = key.replace(f"layers.{layer_idx}.", "")
                layer_weights[new_key] = value
        
        # Apply transformer block
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x
        )
    
    # Final layer norm
    ln_final_weight = weights["ln_final.weight"]
    x = rmsnorm(d_model, 1e-5, ln_final_weight, x)
    
    # Language model head (output projection)
    lm_head_weight = weights["lm_head.weight"]
    logits = linear(d_model, vocab_size, lm_head_weight, x)
    
    return logits

def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    scale = 1.0 / (1.0 + torch.exp(-in_features))
    return in_features * scale

def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    scale = 1.0 / (torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps))
    normed_features = in_features * scale  # (..., d_model)
    return normed_features * weights
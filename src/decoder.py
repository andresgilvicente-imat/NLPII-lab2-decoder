import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionHead(nn.Module):
    """Single Attention Head with Masking Support.

    This class implements a single attention head which is part of the
    multi-head attention mechanism. It computes the attention for a given
    query, key, and value, with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_k (int): The dimension of the key vectors.
        d_q (int): The dimension of the query vectors.
        d_v (int): The dimension of the value vectors.

    Attributes:
        wq (nn.Linear): Linear layer to project input to query vectors.
        wk (nn.Linear): Linear layer to project input to key vectors.
        wv (nn.Linear): Linear layer to project input to value vectors.
    """

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(AttentionHead, self).__init__()

        # TODO: Initialize the linear layers required for the query, key, and value projections.
        self.wq = nn.Linear(in_features=d_model, out_features=d_q)
        self.wk = nn.Linear(in_features=d_model, out_features=d_k)
        self.wv = nn.Linear(in_features=d_model, out_features=d_v)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights with optional causal mask.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_q).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_v).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor after applying attention.
            Tensor: Attention weights.
        """
        # TODO: Implement the scaled dot-product attention mechanism, now with masking.

        d_k = k.shape[2]

        scale = math.sqrt(d_k)

        if mask is not None:
            weights = torch.softmax(
                torch.bmm(q, torch.transpose(k, dim0=1, dim1=2)).masked_fill(
                    mask=mask == 0, value=float("-inf")
                )
                / scale,
                dim=2,
            )
        else:
            weights = torch.softmax(
                torch.bmm(q, torch.transpose(k, dim0=1, dim1=2)) / scale, dim=2
            )

        output = torch.bmm(weights, v)

        return output, weights

    def forward(self, x, mask=None):
        """Forward pass for the attention head with optional causal mask.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        # TODO: Implement the forward pass for the attention head, now with masking.

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, weights = self.scaled_dot_product_attention(q, k, v, mask)

        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with Masking Support.

    This class implements the multi-head attention mechanism, allowing
    the model to focus on different parts of the input sequence at each layer,
    with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.

    Attributes:
        heads (nn.ModuleList): A list of attention heads.
        output_linear (nn.Linear): Linear layer to project concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()

        # TODO: Define the heads and linear layer

        # we store these to make the division check in the forward
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads

        # d_k = d_q = d_v = fraction (d_model)
        d_k = d_model // num_attention_heads
        d_q, d_v = d_k, d_k

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_k, d_q, d_v) for _ in range(num_attention_heads)]
        )

        self.output_linear = nn.Linear(
            in_features=num_attention_heads * d_v, out_features=d_model
        )

    def forward(self, hidden_state, mask=None):
        """Forward pass for the multi-head attention layer with optional causal mask.

        Args:
            hidden_state (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # TODO: Implement the forward pass for the multi-head attention layer, now with masking.

        if self.d_model % self.num_attention_heads != 0:
            raise RuntimeError("d_model must be divisible by num_attention_heads")

        heads_outputs = [head(hidden_state, mask) for head in self.heads]
        concatenated_outputs = torch.cat(heads_outputs, dim=2)
        x = self.output_linear(concatenated_outputs)

        return x


class FeedForward(nn.Module):
    """FeedForward module for the Transformer.

    This class implements the feed-forward network used in the Transformer
    model. It consists of two linear layers with a GELU activation in between.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        intermediate_size (int): The dimension of the intermediate layer.

    Attributes:
        linear_1 (nn.Linear): The first linear layer that projects from d_model to intermediate_size.
        linear_2 (nn.Linear): The second linear layer that projects from intermediate_size back to d_model.
        gelu (nn.GELU): GELU activation function applied after the first linear layer.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super(FeedForward, self).__init__()

        # TODO: Define the different layers

        self.linear_1 = nn.Linear(in_features=d_model, out_features=intermediate_size)

        self.linear_2 = nn.Linear(in_features=intermediate_size, out_features=d_model)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # TODO: Implement the forward pass for the feed-forward network

        z0 = self.linear_1(x)
        z1 = self.gelu(z0)
        z2 = self.linear_2(z1)

        return z2


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    This class implements a single layer of the Transformer decoder, consisting
    of a masked multi-head self-attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization before self-attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization before feed-forward network.
        self_attention (MultiHeadAttention): Masked multi-head self-attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerDecoderLayer, self).__init__()

        # TODO: Initialize the sub-layers
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model)
        self.self_attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # TODO: Implement the forward pass for the Transformer decoder layer

        y1 = x + self.self_attention(self.layer_norm_1(x), mask)

        y2 = y1 + self.feed_forward(self.layer_norm_2(y1))

        return y2


class Embeddings(nn.Module):
    """Embeddings module for the Transformer.

    This module combines token embeddings and positional embeddings and applies
    layer normalization.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.

    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for token embeddings.
        position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
        layer_norm (nn.LayerNorm): Layer normalization applied after combining embeddings.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int):
        super(Embeddings, self).__init__()

        # TODO: Define the different layers of the embeddings

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings, embedding_dim=d_model
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to combine token and positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined and normalized embeddings of shape (batch_size, seq_len, d_model).
        """
        # TODO: Implement the forward pass for the embeddings
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device)
        embeddings = self.layer_norm(
            self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        )

        return embeddings


class TransformerDecoder(nn.Module):
    """Transformer Decoder.

    This class implements the decoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer decoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer decoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer decoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        d_model: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
    ):
        super(TransformerDecoder, self).__init__()

        # TODO: Define the embeddings layer and the decoder layers
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_attention_heads, intermediate_size)
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # TODO: Implement the forward pass for the Transformer decoder
        x = self.embeddings(input_ids)

        batch_size, seq_len = input_ids.shape

        lower_triangular_mask = torch.ones(
            size=(batch_size, seq_len, seq_len), dtype=torch.bool, device=x.device
        ).tril()  # we want to keep the diagonal with True values

        for layer in self.layers:
            x = layer(x, lower_triangular_mask)

        return x


class TransformerForLanguageModeling(nn.Module):
    """Transformer model with a language modeling head for text generation.

    Args:
        vocab_size (int): Vocabulary size.
        max_position_embeddings (int): Maximum number of position embeddings.
        d_model (int): Hidden size of the Transformer model.
        num_attention_heads (int): Number of attention heads in each decoder layer.
        intermediate_size (int): Intermediate size of the feed-forward network.
        num_hidden_layers (int): Number of Transformer decoder layers.

    Attributes:
        transformer_decoder (TransformerDecoder): The Transformer decoder.
        lm_head (nn.Linear): Linear layer mapping hidden states to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        d_model: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
    ):
        super(TransformerForLanguageModeling, self).__init__()

        # TODO: Define the Transformer decoder and the language modeling head
        self.transformer_decoder = TransformerDecoder(
            vocab_size,
            max_position_embeddings,
            d_model,
            num_attention_heads,
            intermediate_size,
            num_hidden_layers,
        )

        self.lm_head = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer model with language modeling head.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """

        # TODO: Implement the forward pass for the Transformer model

        transformer_decoder_outputs = self.transformer_decoder(input_ids)
        logits = self.lm_head(transformer_decoder_outputs)

        return logits

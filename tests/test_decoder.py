import math
import torch
import torch.nn.functional as F
import pytest

from src.decoder import (
    AttentionHead,
    MultiHeadAttention,
    FeedForward,
    Embeddings,
    TransformerDecoderLayer,
    TransformerDecoder,
    TransformerForLanguageModeling,
)


@pytest.fixture
def seed():
    torch.manual_seed(0)


def _find_submodule_by_type(module, cls):
    for child in module.children():
        if isinstance(child, cls):
            return child
    raise AssertionError(f"No submodule of type {cls.__name__} found inside {module}")


# -------- AttentionHead --------

@pytest.mark.order(1)
def test_attention_head():
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_k = d_q = d_v = 8

    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)

    attention_head = AttentionHead(d_model, d_k, d_q, d_v)
    output = attention_head(x, mask)

    assert output.shape == (batch_size, seq_len, d_v), "Output shape mismatch in AttentionHead"


@pytest.mark.order(2)
def test_attention_head_applies_sqrt_dk_scaling():
    # Craft q/k directly (bypassing wq/wk) so the raw dot products are large
    # enough that scaling vs. not scaling by sqrt(d_k) gives clearly
    # different softmax distributions.
    d = 8
    attention_head = AttentionHead(d, d, d, d)
    b, t = 1, 2
    q = torch.zeros(b, t, d)
    k = torch.zeros(b, t, d)
    q[0, 0, 0], k[0, 0, 0] = 4.0, 4.0
    q[0, 1, 1], k[0, 1, 1] = 4.0, 4.0
    v = torch.randn(b, t, d)

    _, weights = attention_head.scaled_dot_product_attention(q, k, v, mask=None)

    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    expected_weights = F.softmax(expected_scores, dim=-1)
    assert torch.allclose(weights, expected_weights, atol=1e-5)


@pytest.mark.order(3)
def test_attention_head_causal_mask_blocks_future_positions():
    # A correct causal mask must zero out attention to strictly-future
    # positions (weights[..., i, j] == 0 for j > i) while still leaving each
    # row a valid probability distribution over the allowed positions. This
    # catches both an inverted mask (mask==1 treated as "block") and a mask
    # applied to the wrong triangle.
    torch.manual_seed(0)
    b, t, d = 2, 5, 8
    attention_head = AttentionHead(d, d, d, d)
    q = torch.randn(b, t, d)
    k = torch.randn(b, t, d)
    v = torch.randn(b, t, d)
    mask = torch.tril(torch.ones(t, t)).unsqueeze(0).repeat(b, 1, 1)

    _, weights = attention_head.scaled_dot_product_attention(q, k, v, mask)

    future_positions = torch.triu(torch.ones(t, t), diagonal=1).bool().unsqueeze(0).expand(b, t, t)
    assert torch.allclose(
        weights[future_positions], torch.zeros_like(weights[future_positions]), atol=1e-6
    )

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# -------- MultiHeadAttention --------

@pytest.mark.order(4)
def test_multi_head_attention():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)

    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    output = multi_head_attention(x, mask)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch in MultiHeadAttention"


@pytest.mark.order(5)
def test_multi_head_attention_heads_are_independent_modules():
    # A common bug is building the head list with `[AttentionHead(...)] * n`,
    # which repeats the SAME module instance instead of creating independent
    # heads. Shapes still come out right, so only checking identities/weights
    # catches it.
    torch.manual_seed(0)
    num_attention_heads = 2
    multi_head_attention = MultiHeadAttention(d_model=8, num_attention_heads=num_attention_heads)
    heads = list(multi_head_attention.heads)
    assert len(heads) == num_attention_heads
    assert len(heads) == len(set(id(h) for h in heads))
    assert not torch.equal(heads[0].wq.weight, heads[1].wq.weight)


# -------- FeedForward --------

@pytest.mark.order(6)
def test_feed_forward():
    batch_size = 2
    seq_len = 4
    d_model = 8
    intermediate_size = 16

    x = torch.rand(batch_size, seq_len, d_model)
    feed_forward = FeedForward(d_model, intermediate_size)
    output = feed_forward(x)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch in FeedForward"


@pytest.mark.order(7)
def test_feedforward_uses_gelu_activation():
    # Force linear_1 and linear_2 to be the identity (square d_model ==
    # intermediate_size) so the only thing left in the forward pass is the
    # non-linearity itself. A ReLU (or any activation other than GELU) will
    # diverge from the expected values for negative inputs.
    d = 4
    ff = FeedForward(d_model=d, intermediate_size=d)
    with torch.no_grad():
        ff.linear_1.weight.copy_(torch.eye(d))
        ff.linear_1.bias.zero_()
        ff.linear_2.weight.copy_(torch.eye(d))
        ff.linear_2.bias.zero_()

    x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    y = ff(x)
    expected = F.gelu(x)
    assert torch.allclose(y, expected, atol=1e-5)


# -------- Embeddings --------

@pytest.mark.order(8)
def test_embeddings_use_both_token_and_position_information():
    # Forgetting to add either the token or the position embeddings still
    # yields the right shape and doesn't crash, so it goes unnoticed unless
    # we check that both actually influence the output.
    embeddings = Embeddings(vocab_size=50, max_position_embeddings=10, d_model=8)
    b, t = 1, 5

    same_token_every_position = torch.zeros(b, t, dtype=torch.long)
    y_same_token = embeddings(same_token_every_position)
    assert not torch.allclose(y_same_token[:, 0, :], y_same_token[:, 1, :])

    different_token_at_pos0 = same_token_every_position.clone()
    different_token_at_pos0[:, 0] = 1
    y_different_token = embeddings(different_token_at_pos0)
    assert not torch.allclose(y_different_token[:, 0, :], y_same_token[:, 0, :])

    # LayerNorm should produce ~zero mean along the last dim.
    mean = y_same_token.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


# -------- TransformerDecoderLayer --------

@pytest.mark.order(9)
def test_transformer_decoder_layer():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    intermediate_size = 16

    x = torch.rand((batch_size, seq_len, d_model))
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

    decoder_layer = TransformerDecoderLayer(d_model, num_heads, intermediate_size)

    output_with_mask = decoder_layer(x, mask)
    output_no_mask = decoder_layer(x, None)

    assert not torch.allclose(output_with_mask, output_no_mask), "Mask is not being applied; outputs are identical."


@pytest.mark.order(10)
def test_decoder_layer_residual_stream_preserves_input():
    # The feed-forward sub-layer must add its residual to the OUTPUT of the
    # attention sub-layer (x after the first residual add), not to the raw
    # attention output again. Zeroing both sub-layers isolates the residual
    # bookkeeping: with correct residuals the layer must act as identity;
    # a layer that re-adds the attention output instead of x will drop the
    # original input entirely and return zeros here.
    #
    # The attention/feed-forward submodules are located by type rather than
    # by a fixed attribute name (some correct submissions call the attention
    # attribute `attention`, others `self_attention`).
    torch.manual_seed(0)
    d_model, num_heads, intermediate_size = 8, 2, 16
    layer = TransformerDecoderLayer(d_model, num_heads, intermediate_size)

    attention_module = _find_submodule_by_type(layer, MultiHeadAttention)
    feed_forward_module = _find_submodule_by_type(layer, FeedForward)
    attention_module.forward = lambda h, mask=None: torch.zeros_like(h)
    feed_forward_module.forward = lambda h: torch.zeros_like(h)

    b, t, d = 2, 5, d_model
    x = torch.randn(b, t, d)
    mask = torch.tril(torch.ones(t, t)).unsqueeze(0).repeat(b, 1, 1)
    y = layer(x, mask)
    assert torch.allclose(y, x, atol=1e-6)


# -------- TransformerDecoder --------

@pytest.mark.order(11)
def test_transformer_decoder():
    batch_size = 2
    seq_len = 4
    vocab_size = 50
    max_position_embeddings = 10
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder = TransformerDecoder(vocab_size, max_position_embeddings, d_model,
                                num_heads, intermediate_size, num_layers)
    output = decoder(input_ids)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch in TransformerDecoder"

    # The stack of decoder layers must actually be used: a decoder that
    # (bug) skips straight from embeddings to the output would still have
    # the right shape here.
    emb = decoder.embeddings(input_ids)
    assert not torch.equal(output, emb)


@pytest.mark.order(12)
def test_transformer_decoder_layers_match_num_hidden_layers_and_are_independent():
    # Two common bugs: hardcoding a single layer regardless of
    # num_hidden_layers, or building the layer list with
    # `[TransformerDecoderLayer(...)] * n` (same instance repeated).
    num_hidden_layers = 3
    decoder = TransformerDecoder(
        vocab_size=50,
        max_position_embeddings=10,
        d_model=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
    )
    layers = list(decoder.layers)
    assert len(layers) == num_hidden_layers
    assert len(set(id(layer) for layer in layers)) == num_hidden_layers


@pytest.mark.order(13)
def test_transformer_decoder_causal_mask_blocks_future_tokens():
    # The defining property of a decoder: changing a later token must never
    # change the hidden state at an earlier position. This catches a missing
    # mask, an inverted mask, or an off-by-one in how it's built -- none of
    # which necessarily break shape or the weaker "output changes with the
    # mask" check.
    torch.manual_seed(0)
    vocab_size, max_position_embeddings, d_model = 50, 10, 8
    num_heads, intermediate_size, num_layers = 2, 16, 2

    decoder = TransformerDecoder(
        vocab_size, max_position_embeddings, d_model, num_heads, intermediate_size, num_layers
    )
    decoder.eval()

    b, t = 2, 6
    input_ids = torch.randint(0, vocab_size, (b, t))
    modified_ids = input_ids.clone()
    modified_ids[:, -1] = (modified_ids[:, -1] + 1) % vocab_size

    with torch.no_grad():
        out_original = decoder(input_ids)
        out_modified = decoder(modified_ids)

    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-6)


# -------- TransformerForLanguageModeling --------

@pytest.mark.order(14)
def test_transformer_for_language_modeling():
    batch_size = 2
    seq_len = 4
    vocab_size = 50
    max_position_embeddings = 10
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = TransformerForLanguageModeling(vocab_size, max_position_embeddings, d_model,
                                            num_heads, intermediate_size, num_layers)
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch in TransformerForLanguageModeling"

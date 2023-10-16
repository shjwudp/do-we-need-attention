# Architecture

# Hyperparameters

channel mixer hp:
- d_model: embedding dimension, notion as $D_e$ in paper

token mixer hp:
- order: depth of hyena FFN, notion as $N$ in paper
- l_max: max length of sequence, notion as $L$ in paper
- filter_order: width of MLP in hyena filter, notion as $D$ in paper

```
d_model: 768
d_inner: 3072
n_layer: 12
vocab_size: ${dataset.vocab_size}
max_position_embeddings: 0
resid_dropout: 0.0
embed_dropout: 0.1
layer_norm_epsilon: 0.00001
pad_vocab_size_multiple: 1
layer:
    _name_: "hyena"
    l_max: ${eval:'${dataset.1} + 2'}
    order: 4
    filter_order: 64
    num_heads: 1
    inner_factor: 1
    num_blocks: 1
    outer_mixing: False
    drop_rate: 0.15
    filter_dropout: 0.0
    filter_cls: 'hyena-filter'
    post_order_ffn: False
    short_filter_order: 3
    activation_type: "id"
    return_state: False
    filter_args:
        emb_dim: 4 # dim of input to MLP, augments with positional encoding
        w: 14  # frequency of periodic activations (note filter configs say 1, default in object is 10, and final hydra filter indicates 10)
        use_bias: True
        num_inner_mlps: 2
        normalized: False
```

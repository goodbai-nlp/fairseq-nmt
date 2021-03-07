# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, TransformerEncoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class FeatLstmEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.feat_layer = TransformerEncoderLayer(args)
        self.dropout = args.dropout
        # feature fusion network, i.e., lstm
        self.fusion_net = nn.LSTMCell(self.embed_dim, self.embed_dim)
        
        self.h = None
        self.c = None

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, prev_state, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        

        # extract features 
        x = self.feat_layer(x, encoder_padding_mask=encoder_padding_mask)

        # fuse features
        seq_len, bsz, embed_dim = x.size()

        x = torch.reshape(x, (seq_len*bsz,embed_dim))
        if not prev_state:
            h, c = self.fusion_net(x)
        else:
            h, c = self.fusion_net(x, prev_state)
        x = torch.reshape(h, (seq_len, bsz, embed_dim))

        return x, (h,c)

    # def forward_tf_layer(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
    #     """
    #     Args:
    #         x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
    #         encoder_padding_mask (ByteTensor): binary ByteTensor of shape
    #             `(batch, seq_len)` where padding elements are indicated by ``1``.
    #         attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
    #             where `tgt_len` is the length of output and `src_len` is the
    #             length of input, though here both are equal to `seq_len`.
    #             `attn_mask[tgt_i, src_j] = 1` means that when calculating the
    #             embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
    #             useful for strided self-attention.

    #     Returns:
    #         encoded output of shape `(seq_len, batch, embed_dim)`
    #     """
    #     # anything in original attn_mask = 1, becomes -1e8
    #     # anything in original attn_mask = 0, becomes 0
    #     # Note that we cannot use -inf here, because at some edge cases,
    #     # the attention weight (before softmax) for some padded element in query
    #     # will become -inf, which results in NaN in model parameters
    #     if attn_mask is not None:
    #         attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

    #     residual = x
    #     if self.normalize_before:
    #         x = self.self_attn_layer_norm(x)
    #     x, _ = self.self_attn(
    #         query=x,
    #         key=x,
    #         value=x,
    #         key_padding_mask=encoder_padding_mask,
    #         need_weights=False,
    #         attn_mask=attn_mask,
    #     )
    #     x = self.dropout_module(x)
    #     x = self.residual_connection(x, residual)
    #     if not self.normalize_before:
    #         x = self.self_attn_layer_norm(x)

    #     residual = x
    #     if self.normalize_before:
    #         x = self.final_layer_norm(x)
    #     x = self.activation_fn(self.fc1(x))
    #     x = self.activation_dropout_module(x)
    #     x = self.fc2(x)
    #     x = self.dropout_module(x)
    #     x = self.residual_connection(x, residual)
    #     if not self.normalize_before:
    #         x = self.final_layer_norm(x)
    #     return x



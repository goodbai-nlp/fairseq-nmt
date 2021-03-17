# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, TransformerEncoderLayer, LightweightConv
# from fairseq.models.lightconv import LightConvEncoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from torch import Tensor


class LightConvEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, kernel_size=0):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.conv_dim = args.encoder_conv_dim
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        if args.encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.encoder_conv_type == "lightweight":
            self.conv = LightweightConv(
                self.conv_dim,
                kernel_size,
                padding_l=padding_l,
                weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads,
                weight_dropout=args.weight_dropout,
            )
        elif args.encoder_conv_type == "dynamic":
            self.conv = DynamicConv(
                self.conv_dim,
                kernel_size,
                padding_l=padding_l,
                weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads,
                weight_dropout=args.weight_dropout,
            )
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.relu_dropout_module = FairseqDropout(
            args.relu_dropout, module_name=self.__class__.__name__
        )
        self.input_dropout_module = FairseqDropout(
            args.input_dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_module(x)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def extra_repr(self):
        return (
            "dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}".format(
                self.dropout_module.p,
                self.relu_dropout_module.p,
                self.input_dropout_module.p,
                self.normalize_before,
            )
        )

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
        x = torch.reshape(x, (seq_len, bsz, embed_dim))

        return x, (h,c)


class MultiFeatsEncoderLayer(nn.Module):
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
        self.feat0 = FeedforwardLayer(self.embed_dim, args.encoder_ffn_embed_dim)
        self.feat1 = SelfAttentionLayer(self.embed_dim, args.encoder_attention_heads)
        self.dropout = args.dropout
        # feature fusion network, i.e., lstm
        self.fusion_net = nn.LSTMCell(self.embed_dim*2, self.embed_dim)
        

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
        
        # if type(x) is tuple:
        #     x0 = x
        #     x1 = x
        # else:
        #     x0,x1 = x
        if x.size()[-1] == self.embed_dim:
            x0 = x
            x1 = x
        else:
            x0 = x[:,:,:self.embed_dim]
            x1 = x[:,:,self.embed_dim:]
        seq_len, bsz, embed_dim = x0.size()
        # extract features 

        x0 = self.feat0(x0)
        x1 = self.feat1(x1, encoder_padding_mask, attn_mask)

        # fuse features     
        x = torch.cat((x0,x1), dim=-1)
        x = torch.reshape(x, (seq_len*bsz,embed_dim*2))
        if not prev_state:
            h, c = self.fusion_net(x)
        else:
            h, c = self.fusion_net(x, prev_state)
        x = torch.reshape(x, (seq_len, bsz, embed_dim*2))

        return x, (h,c)

class MultiFeatsGenEncoderLayer(nn.Module):
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
        self.feat0 = LightConvEncoderLayer(args, kernel_size=args.conv_kernel_size)
        self.feat1 = TransformerEncoderLayer(args)
        self.dropout = args.dropout
        # feature fusion network, i.e., lstm
        self.fusion_net = nn.LSTMCell(self.embed_dim*2, self.embed_dim)
        

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
        
        # if type(x) is tuple:
        #     x0 = x
        #     x1 = x
        # else:
        #     x0,x1 = x
        if x.size()[-1] == self.embed_dim:
            x0 = x
            x1 = x
        else:
            x0 = x[:,:,:self.embed_dim]
            x1 = x[:,:,self.embed_dim:]
        seq_len, bsz, embed_dim = x0.size()
        # extract features 

        x0 = self.feat0(x0, encoder_padding_mask)
        x1 = self.feat1(x1, encoder_padding_mask=encoder_padding_mask)

        # fuse features     
        x = torch.cat((x0,x1), dim=-1)
        x = torch.reshape(x, (seq_len*bsz,embed_dim*2))
        if not prev_state:
            h, c = self.fusion_net(x)
        else:
            h, c = self.fusion_net(x, prev_state)
        x = torch.reshape(x, (seq_len, bsz, embed_dim*2))

        return x, (h,c)

class FeedforwardLayer(nn.Module):
    def __init__(self, hidden_size, ffn_dim, dropout=0.1, act_func='relu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.activation_fn = utils.get_activation_fn(act_func)
        self.dropout_module = FairseqDropout(
            float(self.dropout), module_name=self.__class__.__name__
        )
        self.fc1 = nn.Linear(self.hidden_size, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.hidden_size)


    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, act_func='relu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout  = dropout
        self.activation_fn = utils.get_activation_fn(act_func)
        self.attention = MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            self_attention=True
        )
    
    def forward(self, x, encoder_padding_mask, attn_mask):

        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.activation_fn(x)
        return x

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m        
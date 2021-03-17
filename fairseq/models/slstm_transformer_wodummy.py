# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq import options as fairseq_options
from fairseq import utils as fairseq_utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.data.dictionary import Dictionary
from fairseq.models import FairseqEncoder
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from fairseq.models.fairseq_model import FairseqEncoderDecoderModel
from fairseq.models.transformer import TransformerEncoder, Linear
from fairseq.modules.adaptive_softmax import AdaptiveSoftmax
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    SLSTMWoDummy,
    FairseqDropout,
    TransformerDecoderLayer
)
from torch import Tensor
BufferType = Tuple[torch.Tensor, torch.Tensor]
DEFAULT_MAX_SOURCE_POSITIONS = 1024


@register_model("slstm_tf_wodummy")
class SlstmWodummyTransformerModel(FairseqEncoderDecoderModel):
    """Transformer with LSTM based decoder.

    This model takes advantage of the performance of the transformer
    encoder while scaling linearly in decoding speed and memory against
    generated sequence length.

    Similar to the standard transformer, this model takes advantage
    of multi-head attention to attend on encoder outputs.
    However, instead of using multi-head self-attention, this model uses
    a LSTM blocks instead.

    The model provides the following command-line arguments

    .. argparse::
        :ref: fairseq.models.lstm_decode_transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--activation-fn",
            choices=fairseq_utils.get_available_activation_fns(),
            help="activation function to use",
        )

        parser.add_argument("--dropout", type=float, metavar="D", help="dropout probability")
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )

        parser.add_argument(
            "--encoder-embed-path",
            type=str,
            metavar="STR",
            help="path to pre-trained encoder embedding",
        )
        parser.add_argument(
            "--encoder-embed-dim", type=int, metavar="N", help="encoder embedding dimension"
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument("--encoder-layers", type=int, metavar="N", help="num encoder layers")
        parser.add_argument(
            "--encoder-attention-heads", type=int, metavar="N", help="num encoder attention heads"
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )

        parser.add_argument(
            "--decoder-embed-path",
            type=str,
            metavar="STR",
            help="path to pre-trained decoder embedding",
        )
        parser.add_argument(
            "--decoder-embed-dim", type=int, metavar="N", help="decoder embedding dimension"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="number of decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads", type=int, metavar="N", help="num decoder attention heads"
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )

        parser.add_argument(
            "--adaptive-softmax-cutoff",
            metavar="EXPR",
            help="comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion",
        ),
        parser.add_argument(
            "--adaptive-softmax-dropout",
            type=float,
            metavar="D",
            help="sets adaptive softmax dropout for the tail projections",
        )

        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--share-all-embeddings",
            action="store_true",
            help="share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)",
        )

        parser.add_argument(
            "--layernorm-embedding", action="store_true", help="add layernorm to embedding"
        )
        parser.add_argument(
            "--no-scale-embedding", action="store_true", help="if True, dont scale embeddings"
        )

        # parser.add_argument('--use-slstm', default=False, action='store_true',
        #                     help='if True, use s-lstm encoder')
        parser.add_argument(
            "--kernel-size", type=int, metavar="N", default=1, help="kernel size for s-lstm layer"
        )
        parser.add_argument(
            "--temperature", type=float, default=1.0, help="temperature for dummynode softmax"
        )
        parser.add_argument(
            "--use-layerwise-global",
            default=False,
            action="store_true",
            help="use global hidden and cell for decoder init",
        )
        parser.add_argument(
            "--use-last-global",
            default=False,
            action="store_true",
            help="use last global hidden and cell for decoder init",
        )
        parser.add_argument(
            "--merge_layer",
            default=False,
            action="store_true",
            help="merge layers for decoding",
        )
        parser.add_argument(
            "--mask_dummy_for_fgate",
            default=False,
            action="store_true",
            help="reset dummy hiddens for calc fgates",
        )
        parser.add_argument(
            "--ffoncell",
            default=False,
            action="store_true",
            help="use ff on cell",
        )
        parser.add_argument(
            "--layer-wise-attention",
            default=False,
            action="store_true",
            help="use layer-wise cross attention in decoder",
        )
        # parser.add_argument('--use-global-feeding', default=False, action='store_true',
        #                     help='use global hidden and cell for decoder init')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = fairseq_utils.parse_embedding(path)
                fairseq_utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            if args.decoder_embed_path and args.decoder_embed_path != args.encoder_embed_path:
                raise ValueError(
                    "--share-all-embeddings not compatible with " "--decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SLSTMEncoder(args=args, dictionary=src_dict, embed_tokens=embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


class SLSTMEncoder(FairseqEncoder):
    """SLSTM encoder."""

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop
        self.kernel_size = args.kernel_size

        self.padding_idx = embed_tokens.padding_idx
        embed_dim = embed_tokens.embedding_dim
        self.output_units = embed_dim
        self.use_layerwise_global = args.use_layerwise_global
        self.use_last_global = args.use_last_global
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        # print(args)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        # self.embed_positions = None

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)
        self.layers = self.build_encoder_layer(args)

        # self.layers = nn.ModuleList([])
        # self.layers.extend(
        #     [self.build_encoder_layer(args) for i in range(1)]
        # )
        self.num_layers = args.encoder_layers

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def build_encoder_layer(self, args):
        # return SLSTMFast(args)
        return SLSTMWoDummy(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            # print('use position emb')
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            # print('use layernorm emb')
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()
        # if self.layer_wise_attention:
        #     return_all_hiddens = True
        # else:
        #     return_all_hiddens = False

        # embed tokens
        x, encoder_embedding = self.forward_embedding(src_tokens)
        # x = self.embed_tokens(src_tokens)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x: B x T x C
        encoder_padding_mask = src_tokens.eq(self.padding_idx)      # [bsz, T]
        # if encoder_padding_mask.any():
        #     print('src_token', src_tokens, src_tokens.size())
        #     print('padding_mask', encoder_padding_mask, encoder_padding_mask.size())
        #     exit()
        # encoder_states = [] if return_all_hiddens else None
        x, c, g_h, g_c, all_mem = self.layers(x, encoder_padding_mask, self.num_layers, src_lengths)

        # T x B x C
        x = x.transpose(0, 1)
        c = c.transpose(0, 1)
        all_mem = all_mem.transpose(1, 2)  # [num_layer, T, B, C]
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            # if return_all_hiddens:
            #     encoder_states[-1] = x

        assert list(x.size()) == [seqlen, bsz, self.output_units]

        

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "src_tokens": [],
            "src_lengths": [],
        }



    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out["encoder_out"]
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [encoder_out["encoder_padding_mask"][0].index_select(
                0, new_order
            )]
        # if encoder_out["all_encoder_out"] is not None:
        #     encoder_out["all_encoder_out"] = encoder_out["all_encoder_out"].index_select(
        #         2, new_order
        #     )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions




class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fairseq_utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if fairseq_utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m




@register_model_architecture("slstm_tf_wodummy", "slstm_tf_wodummy")
def base_architecture(args):
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)

    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.dec_layernorm_embedding = getattr(args, "dec_layernorm_embedding", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.quant_noise_pq = 0
    args.encoder_layerdrop = 0
    args.no_token_positional_embeddings = False




@register_model_architecture("slstm_tf_wodummy", "slstm_tf_wodummy_iwslt")
def slstm_tlstm_iwslt(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    base_architecture(args)

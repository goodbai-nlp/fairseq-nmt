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
from fairseq.models.fairseq_incremental_decoder import \
    FairseqIncrementalDecoder
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
    SLSTMFast,
    SLSTMFast4,
)
from torch import Tensor
BufferType = Tuple[torch.Tensor, torch.Tensor]
DEFAULT_MAX_SOURCE_POSITIONS = 1024


@register_model('slstm_tf_like')
class LSTMDecodeTransformerModel(FairseqEncoderDecoderModel):
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
            '--activation-fn',
            choices=fairseq_utils.get_available_activation_fns(),
            help='activation function to use')

        parser.add_argument(
            '--dropout', type=float, metavar='D',
            help='dropout probability')
        parser.add_argument(
            '--attention-dropout', type=float, metavar='D',
            help='dropout probability for attention weights')
        parser.add_argument(
            '--activation-dropout', '--relu-dropout', type=float, metavar='D',
            help='dropout probability after activation in FFN.')

        parser.add_argument(
            '--encoder-embed-path', type=str, metavar='STR',
            help='path to pre-trained encoder embedding')
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='encoder embedding dimension')
        parser.add_argument(
            '--encoder-ffn-embed-dim', type=int, metavar='N',
            help='encoder embedding dimension for FFN')
        parser.add_argument(
            '--encoder-layers', type=int, metavar='N',
            help='num encoder layers')
        parser.add_argument(
            '--encoder-attention-heads', type=int, metavar='N',
            help='num encoder attention heads')
        parser.add_argument(
            '--encoder-normalize-before', action='store_true',
            help='apply layernorm before each encoder block')
        parser.add_argument(
            '--encoder-learned-pos', action='store_true',
            help='use learned positional embeddings in the encoder')

        parser.add_argument(
            '--decoder-embed-path', type=str, metavar='STR',
            help='path to pre-trained decoder embedding')
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='decoder embedding dimension')
        parser.add_argument(
            '--decoder-ffn-embed-dim', type=int, metavar='N',
            help='decoder embedding dimension for FFN')
        parser.add_argument(
            '--decoder-layers', type=int, metavar='N',
            help='number of decoder layers')
        parser.add_argument(
            '--decoder-attention-heads', type=int, metavar='N',
            help='num decoder attention heads')
        parser.add_argument(
            '--decoder-out-embed-dim', type=int, metavar='N',
            help='decoder output embedding dimension')
        parser.add_argument(
            '--decoder-normalize-before', action='store_true',
            help='apply layernorm before each decoder block')

        parser.add_argument(
            '--adaptive-softmax-cutoff', metavar='EXPR',
            help='comma separated list of adaptive softmax cutoff points. '
            'Must be used with adaptive_loss criterion'),
        parser.add_argument(
            '--adaptive-softmax-dropout', type=float, metavar='D',
            help='sets adaptive softmax dropout for the tail projections')

        parser.add_argument(
            '--share-decoder-input-output-embed', action='store_true',
            help='share decoder input and output embeddings')
        parser.add_argument(
            '--share-all-embeddings', action='store_true',
            help='share encoder, decoder and output embeddings'
            ' (requires shared dictionary and embed dim)')

        parser.add_argument(
            '--layernorm-embedding', action='store_true',
            help='add layernorm to embedding')
        parser.add_argument(
            '--no-scale-embedding', action='store_true',
            help='if True, dont scale embeddings')

        parser.add_argument('--use-slstm', default=False, action='store_true',
                            help='if True, use s-lstm encoder')
        parser.add_argument('--use-slstm-fast', default=False, action='store_true',
                            help='if True, use parallel s-lstm encoder')
        parser.add_argument('--use-slstm-fast2', default=False, action='store_true',
                            help='if True, use parallel s-lstm encoder')
        parser.add_argument('--use-slstm-fast3', default=False, action='store_true',
                            help='if True, use parallel s-lstm encoder')
        parser.add_argument('--use-slstm-fast4', default=False, action='store_true',
                            help='if True, use parallel s-lstm encoder')
        parser.add_argument('--use-slstm-fast5', default=False, action='store_true',
                            help='if True, use parallel s-lstm encoder')
        parser.add_argument('--use_slstm_dec', default=False, action='store_true',
                            help='if True, use s-lstm decoder')
        parser.add_argument('--dummy_ff', default=False, action='store_true',
                            help='if True, use dummy_ff layer')
        parser.add_argument('--share_slstm', default=False, action='store_true',
                            help='if True, shared slstm layer among multiheads')
        parser.add_argument('--slstm_heads', default=1, type=int, metavar='N',
                            help='S-LSTM encoder heads')
        parser.add_argument('--slstm_head_dim', default=512, type=int, metavar='N',
                            help='S-LSTM encoder head dimension')
        parser.add_argument('--kernel-size', type=int, metavar='N', default=1,
                            help='kernel size for s-lstm layer')
        parser.add_argument('--temperature', type=float, default=1.0,
                            help='temperature for dummynode softmax')
        parser.add_argument('--base_layer', type=int, default=6,
                            help='base/avg layers')
        parser.add_argument('--std_layer', type=int, default=2,
                            help='std number of layers')
        parser.add_argument('--sigmoid_alpha', type=float, default=0.08,
                            help='alpha of sigmoid func')
        parser.add_argument('--use-global', default=False, action='store_true',
                            help='use global hidden and cell for decoder init')
        parser.add_argument('--use-last-global', default=False, action='store_true',
                            help='use global hidden and cell for decoder init')
        parser.add_argument('--use-global-feeding', default=False, action='store_true',
                            help='use global hidden and cell for decoder init')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        if getattr(args, 'max_source_positions', None) is None:
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
                raise ValueError(
                    '--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            if (
                args.decoder_embed_path and
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    '--share-all-embeddings not compatible with '
                    '--decoder-embed-path'
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
        return LSTMDecodeTransformerDecoder(
            args=args,
            dictionary=tgt_dict,
            embed_tokens=embed_tokens,
            embed_dim=args.decoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            encoder_embed_dim=args.encoder_embed_dim,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_attention_heads,
            activation_fn=args.activation_fn,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            share_input_output_embed=args.share_decoder_input_output_embed,
            normalize_before=args.decoder_normalize_before,
            no_encoder_attn=False,
            adaptive_softmax_cutoff=args.adaptive_softmax_cutoff,
            layernorm_embedding=args.layernorm_embedding,
            no_scale_embedding=args.no_scale_embedding,
        )


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
        self.use_global = args.use_global
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
        return SLSTMFast4(args)
        # return SLSTM(args)

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
        if self.layer_wise_attention:
            return_all_hiddens = True
        
        # embed tokens    
        x, encoder_embedding = self.forward_embedding(src_tokens)
        # x = self.embed_tokens(src_tokens)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x: B x T x C 
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # if encoder_padding_mask.any():
        #     print('src_token', src_tokens, src_tokens.size())
        #     print('padding_mask', encoder_padding_mask, encoder_padding_mask.size())
        #     exit()
        # encoder_states = [] if return_all_hiddens else None
        x, c, g_h, g_c, _ = self.layers(x, encoder_padding_mask, self.num_layers, src_lengths)

        # T x B x C
        x = x.transpose(0, 1)
        c = c.transpose(0, 1)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            # if return_all_hiddens:
            #     encoder_states[-1] = x

        assert list(x.size()) == [seqlen, bsz, self.output_units]
        # num_layer x B x C
        assert list(g_h.size()) == [self.num_layers, bsz, self.output_units]

        if self.use_global:
            # print('using layerwise global')
            final_hiddens = g_h
            final_cells = g_c
        elif self.use_last_global:
            # print('using last global node')
            final_hiddens = g_h[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
            final_cells = g_c[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:
            # print("using zeros initialization")
            final_hiddens = torch.zeros((self.num_layers, x.size(1), x.size(2))).to(x)
            final_cells = torch.zeros((self.num_layers, x.size(1), x.size(2))).to(x)
        # pack embedded source tokens into a PackedSequence
        # packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class LSTMDecodeTransformerDecoder(FairseqIncrementalDecoder):
    """Multihead attention decoder with LSTM instead of self-attn."""

    def __init__(
        self,
        args: None,
        dictionary: Dictionary,
        embed_tokens: nn.Embedding,
        embed_dim: int = 512,
        ffn_embed_dim: int = 512,
        encoder_embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        activation_fn: str = 'relu',
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        share_input_output_embed: bool = False,
        normalize_before: bool = False,
        no_encoder_attn: bool = False,
        adaptive_softmax_cutoff: Optional[str] = None,
        layernorm_embedding: bool = False,
        no_scale_embedding: bool = False,
    ):
        super().__init__(dictionary)

        self.dropout = dropout
        self.share_input_output_embed = share_input_output_embed

        output_embed_dim = input_embed_dim = embed_tokens.embedding_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embed_dim)
        self.padding_idx = embed_tokens.padding_idx
        self.in_proj = None
        if embed_dim != input_embed_dim:
            self.in_proj = Linear(input_embed_dim, embed_dim, bias=False)

        self.out_proj = None
        if embed_dim != output_embed_dim:
            self.out_proj = Linear(embed_dim, output_embed_dim, bias=False)

        self.layers = nn.ModuleList([LSTMTransformerDecoderLayer(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            encoder_embed_dim=encoder_embed_dim,
            num_heads=num_heads,
            activation_fn=activation_fn,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            normalize_before=normalize_before,
            no_encoder_attn=no_encoder_attn
        ) for _ in range(num_layers)])

        self.adaptive_softmax = None
        if adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                fairseq_options.eval_str_list(adaptive_softmax_cutoff),
                dropout=dropout,
            )

        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(
                self.embed_out,
                mean=0,
                std=output_embed_dim ** -0.5
            )

        self.layer_norm = None
        # if normalize_before:
        #     self.layer_norm = LayerNorm(embed_dim)

        self.layernorm_embedding = None
        if layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        
        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.in_proj is not None:
            x = self.in_proj(x)
        if self.embed_positions is not None:
            # print('use position emb')
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            # print('use layernorm emb')
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[dict] = None,
        features_only: bool = False,
        **extra_args,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prev_output_tokens (torch.Tensor):
                Previous decoder outputs of shape `(batch, tgt_len)`.
            encoder_out (EncoderOut, optional):
                Output from encoder. Defaults to None.
            incremental_state (Optional[dict], optional):
                Dictionary caching tensors for efficient sequence generation.
                Defaults to None.
            features_only (bool, optional):
                Only return features without applying output layer.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Tensor of shape `(seq_len, batch, embed_dim)` and
                meta data like atten weights.
        """  # noqa
        assert encoder_out is not None
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """This should be strictly infinite."""
        return int(1e6)

    def extract_features(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[dict] = None,
        features_only: bool = False,
        **unused,
    ):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.in_proj is not None:
            x = self.in_proj(x)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x, _ = self.forward_embedding(prev_output_tokens)
        # BTC -> TBC
        x = x.transpose(0, 1)

        attn = None
        for i, layer in enumerate(self.layers):
            init_states = (encoder_out['encoder_out'][1][i], encoder_out['encoder_out'][2][i])
            x, attn = layer(
                x,
                encoder_out=encoder_out,
                init_states=init_states,
                incremental_state=incremental_state
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # TBC -> BTC
        x = x.transpose(0, 1)

        if self.out_proj is not None:
            x = self.out_proj(x)

        return x, {'attn': [attn]}


@with_incremental_state
class LSTMTransformerDecoderLayer(nn.Module):
    """Multihead attention decoder layer with LSTM instead of self-attn."""

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        encoder_embed_dim: int = 512,
        num_heads: int = 8,
        activation_fn: str = 'relu',
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        normalize_before: bool = False,
        no_encoder_attn: bool = False,
    ):
        super().__init__()

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before

        self.activation_fn = fairseq_utils.get_activation_fn(activation_fn)

        # To be determined if applying LSTM directly is helpful
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False
        )
        self.layer_norm = LayerNorm(embed_dim)

        if no_encoder_attn:
            self.attn = None
            self.attn_layer_norm = None
        else:
            self.attn = MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=encoder_embed_dim,
                vdim=encoder_embed_dim,
                dropout=attention_dropout,
                bias=True,
                encoder_decoder_attention=True,
            )
            self.attn_layer_norm = LayerNorm(embed_dim)

        self.fc1 = Linear(embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)

        self.need_attn = True

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _get_input_buffer(self, incremental_state: dict) -> BufferType:
        return fairseq_utils.get_incremental_state(
            self,
            incremental_state=incremental_state,
            key='rnn_hidden_state'
        )

    def _set_input_buffer(self, incremental_state: dict, buffer: BufferType):
        return fairseq_utils.set_incremental_state(
            self,
            incremental_state=incremental_state,
            key='rnn_hidden_state',
            value=buffer
        )

    def reorder_incremental_state(
        self,
        incremental_state: dict,
        new_order: torch.Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        buffer = self._get_input_buffer(incremental_state)
        if buffer is None:
            return

        h, c = buffer
        new_buffer = (
            h.index_select(1, new_order),
            c.index_select(1, new_order)
        )
        self._set_input_buffer(incremental_state, new_buffer)
        return incremental_state

    def forward(
        self,
        x: torch.Tensor,
        init_states=None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor):
                Tensor of shape `(seq_len, batch, embed_dim)`
            encoder_out (Optional[EncoderOut], optional):
                Encoder output. Defaults to None.
            incremental_state (Optional[dict], optional):
                Dictionary caching tensors for efficient sequence generation.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Tensor of shape `(seq_len, batch, embed_dim)` and
                attention weights.
        """  # noqa
        prev_states = (init_states[0].unsqueeze(0), init_states[1].unsqueeze(0))

        if incremental_state:
            prev_states = self._get_input_buffer(incremental_state)
            if not prev_states:
                prev_states = (init_states[0].unsqueeze(0), init_states[1].unsqueeze(0))    # set init states

        residual = x

        if self.normalize_before:
            x = self.layer_norm(x)
        seqlen, bsz, _ = x.size()
        packed_x = pack_padded_sequence(x, [seqlen] * bsz)
        packed_x, new_states = self.rnn(packed_x, prev_states)
        x, _ = pad_packed_sequence(packed_x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        if not self.normalize_before:
            x = self.layer_norm(x)

        if incremental_state is not None:
            self._set_input_buffer(incremental_state, new_states)

        attn = None
        # if not hasattr(encoder_out, 'encoder_out'):
        #     print(encoder_out)
        #     xx
        assert encoder_out is not None
        if self.attn is not None:
            residual = x
            if self.normalize_before:
                x = self.attn_layer_norm(x)

            x, attn = self.attn(
                query=x,
                key=encoder_out['encoder_out'][0],
                value=encoder_out['encoder_out'][0],
                key_padding_mask=encoder_out['encoder_padding_mask'],
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x += residual
            if not self.normalize_before:
                x = self.attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m



@register_model_architecture(
    'slstm_tf_like', 'slstm_tf_like')
def base_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = \
        getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = \
        getattr(args, "decoder_normalize_before", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_softmax_cutoff = \
        getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = \
        getattr(args, 'adaptive_softmax_dropout', 0)

    args.share_decoder_input_output_embed = \
        getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.dec_layernorm_embedding = getattr(args, "dec_layernorm_embedding", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.quant_noise_pq = 0
    args.encoder_layerdrop = 0
    args.no_token_positional_embeddings = False
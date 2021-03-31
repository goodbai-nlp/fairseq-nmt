# coding:utf-8
import math
import torch
from torch import Tensor, nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import sys
import time
from fairseq.modules import LayerNorm, TransformerEncoderLayer, LightweightConv
from fairseq.modules.quant_noise import quant_noise
from fairseq.incremental_decoding_utils import with_incremental_state
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq import utils as fairseq_utils
from torch.nn import Parameter


class VanillaSLSTMFeat(nn.Module):
    def __init__(self, args):
        # current
        super(VanillaSLSTMFeat, self).__init__()
        self.hidden_size = args.encoder_embed_dim
        self.dropout = args.dropout
        self.kernel_size = args.kernel_size
        # self.l_base = args.base_layer
        # self.l_std = args.std_layer
        # self.s_alpha = args.sigmoid_alpha

        self.norm_gate = nn.Linear(4 * self.hidden_size, 7 * self.hidden_size)


        self.emb_gate_linear = nn.Linear(self.hidden_size, 7 * self.hidden_size, bias=False)

        self.peep_gate_linear = nn.Linear(self.hidden_size, 7 * self.hidden_size, bias=False)


        self.temperature = args.temperature
        # if self.config.HP_gpu:
        #     self.to(self.config.device)

        # reset slstm params
        self.reset_parameters()

        # feat lstm init
        # self.attn_feautre = TransformerEncoderLayer(args)
        self.attn_feautre = MultiheadAttention(
            self.hidden_size,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        # self.attn_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.fc2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.act_func = nn.ReLU(inplace=True)        
        self.ffn_LN = LayerNorm(self.hidden_size)

    def reset_parameters(self):
        self.norm_gates_W = self.norm_gate.weight.t()
        self.norm_gates_b = self.norm_gate.bias
        (
            self.W_fl,
            self.W_fr,
            self.W_fc,
            self.W_fd,
            self.W_i,
            self.W_w,
            self.W_o,
        ) = torch.chunk(
            self.norm_gates_W, 7, dim=1
        )  # [4*H, H]
        sections = [self.hidden_size, 2 * self.hidden_size, self.hidden_size]
        self.Wxf1, self.Whf1, self.Wdf1 = torch.split(self.W_fl, sections, dim=0)  # [H, H]
        self.Wxf2, self.Whf2, self.Wdf2 = torch.split(self.W_fr, sections, dim=0)
        self.Wxf3, self.Whf3, self.Wdf3 = torch.split(self.W_fc, sections, dim=0)
        self.Wxf4, self.Whf4, self.Wdf4 = torch.split(self.W_fd, sections, dim=0)
        self.Wxi, self.Whi, self.Wdi = torch.split(self.W_i, sections, dim=0)
        self.Wxt, self.Wht, self.Wdt = torch.split(self.W_w, sections, dim=0)
        self.Wxo, self.Who, self.Wdo = torch.split(self.W_o, sections, dim=0)
        (
            self.b_fl,
            self.b_fr,
            self.b_fc,
            self.b_fd,
            self.bi,
            self.bw,
            self.bo,
        ) = torch.chunk(self.norm_gates_b, 7, dim=0)
        self.emb_gates_w = self.emb_gate_linear.weight.t()
        (
            self.Wif1,
            self.Wif2,
            self.Wif3,
            self.Wif4,
            self.Wii,
            self.Wit,
            self.Wio,
        ) = torch.chunk(self.emb_gates_w, 7, dim=1)
        self.pip_gates_w = self.peep_gate_linear.weight.t()
        (
            self.Wpf1,
            self.Wpf2,
            self.Wpf3,
            self.Wpf4,
            self.Wpi,
            self.Wpt,
            self.Wpo,
        ) = torch.chunk(self.pip_gates_w, 7, dim=1)

    

        self.initializer = nn.init.normal_
        # self.initializer = nn.init.xavier_normal_
        # self.initializer = nn.init.orthogonal_

        self.initializer(self.Wxf1, mean=0, std=0.1)
        nn.init.normal_(self.Whf1, mean=0, std=0.1)
        nn.init.normal_(self.Wif1, mean=0, std=0.1)
        nn.init.normal_(self.Wdf1, mean=0, std=0.1)
        nn.init.normal_(self.Wpf1, mean=0, std=0.1)

        nn.init.normal_(self.Wxf2, mean=0, std=0.1)
        nn.init.normal_(self.Whf2, mean=0, std=0.1)
        nn.init.normal_(self.Wif2, mean=0, std=0.1)
        nn.init.normal_(self.Wdf2, mean=0, std=0.1)
        nn.init.normal_(self.Wpf2, mean=0, std=0.1)

        nn.init.normal_(self.Wxf3, mean=0, std=0.1)
        nn.init.normal_(self.Whf3, mean=0, std=0.1)
        nn.init.normal_(self.Wif3, mean=0, std=0.1)
        nn.init.normal_(self.Wdf3, mean=0, std=0.1)
        nn.init.normal_(self.Wpf3, mean=0, std=0.1)

        nn.init.normal_(self.Wxf4, mean=0, std=0.1)
        nn.init.normal_(self.Whf4, mean=0, std=0.1)
        nn.init.normal_(self.Wif4, mean=0, std=0.1)
        nn.init.normal_(self.Wdf4, mean=0, std=0.1)
        nn.init.normal_(self.Wpf4, mean=0, std=0.1)

        nn.init.normal_(self.Wxi, mean=0, std=0.1)
        nn.init.normal_(self.Whi, mean=0, std=0.1)
        nn.init.normal_(self.Wii, mean=0, std=0.1)
        nn.init.normal_(self.Wdi, mean=0, std=0.1)
        nn.init.normal_(self.Wpi, mean=0, std=0.1)

        nn.init.normal_(self.Wxt, mean=0, std=0.1)
        nn.init.normal_(self.Wht, mean=0, std=0.1)
        nn.init.normal_(self.Wit, mean=0, std=0.1)
        nn.init.normal_(self.Wdt, mean=0, std=0.1)
        nn.init.normal_(self.Wpt, mean=0, std=0.1)

        nn.init.normal_(self.Wxo, mean=0, std=0.1)
        nn.init.normal_(self.Who, mean=0, std=0.1)
        nn.init.normal_(self.Wio, mean=0, std=0.1)
        nn.init.normal_(self.Wdo, mean=0, std=0.1)
        nn.init.normal_(self.Wpo, mean=0, std=0.1)

        nn.init.normal_(self.bi, mean=0, std=0.1)
        nn.init.normal_(self.bw, mean=0, std=0.1)
        nn.init.normal_(self.bo, mean=0, std=0.1)
        nn.init.normal_(self.b_fl, mean=0, std=0.1)
        nn.init.normal_(self.b_fr, mean=0, std=0.1)
        nn.init.normal_(self.b_fc, mean=0, std=0.1)
        nn.init.normal_(self.b_fd, mean=0, std=0.1)





    def upgrade_state_dict_named(self, state_dict, name):
        pass

    def create_padding_variable(self, *shape):
        if torch.cuda.is_available():
            data = torch.zeros(*shape).to(device=torch.cuda.current_device())
        else:
            data = torch.zeros(*shape)
        var = autograd.Variable(data, requires_grad=False)
        return var

    def get_hidden_states_before(self, padding, hidden_states, step):
        # padding zeros
        # padding = create_padding_variable(self.training, self.config.HP_gpu, (shape[0], step, hidden_size))
        if step < hidden_states.size()[1]:
            # remove last steps
            displaced_hidden_states = hidden_states[:, :-step, :]
            # concat padding
            return torch.cat([padding, displaced_hidden_states], dim=1)
        else:
            return torch.cat([padding] * hidden_states.size()[1], dim=1)

    def get_hidden_states_after(self, padding, hidden_states, step):
        # padding zeros
        # padding = create_padding_variable(self.training, self.config.HP_gpu, (shape[0], step, hidden_size))
        # remove last steps
        if step < hidden_states.size()[1]:
            displaced_hidden_states = hidden_states[:, step:, :]
            # concat padding
            return torch.cat([displaced_hidden_states, padding], dim=1)
        else:
            return torch.cat([padding] * hidden_states.size()[1], dim=1)

    def sum_together(self, ll):
        return sum(ll)


    def attn_feat_extractor(self, x, key_padding_mask, prev_state=None):
        x = x.transpose(0,1) #  -> [seq_len, bsz, H]
        key_padding_mask = torch.squeeze(key_padding_mask, dim=-1)

        # x = self.attn_feautre(x, encoder_padding_mask=key_padding_mask)
        x, _ = self.attn_feautre(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
        )
        x = x.transpose(0,1) # -> [bsz, seq_len, H]
        bsz, seq_len, hidden_size = x.size()

        x = torch.reshape(x, (seq_len*bsz,hidden_size))
        return x

    def forward(self, word_inputs, mask, num_layers, seq_length=None):
        bool_mask = mask
        num_steps = num_layers
        # [bsz, src_len]
        mask_softmax_score = -mask.float() * 1e25


        # filter invalid steps
        sequence_mask = torch.unsqueeze(
            1 - mask.type_as(self.emb_gate_linear.weight), dim=2
        )  # [bsz, src_len, 1]
        # [bsz, src_len, 1]
        sequence_lengths = torch.sum(sequence_mask, dim=1)

        # word_inputs = self.input_norm(word_inputs) #//maybe we can remove this one
        # word_inputs = self.i_drop(word_inputs)
        # filter embedding states

        filtered_word_inputs = word_inputs * sequence_mask  # [bsz, seq_len, H]
        # record shape of the batch
        # [bsz, seq_len, H]
        shape = word_inputs.size()

        # initial embedding states
        # [bsz*seq_len, H]
        embedding_hidden_state = filtered_word_inputs.view(-1, shape[-1])

        # randomly initialize the states
        initial_hidden_states = filtered_word_inputs
        initial_cell_states = filtered_word_inputs


        batch_size, src_len, hidden_size = shape[0], shape[1], shape[2]
        padding_list = [
            self.create_padding_variable((batch_size, step + 1, hidden_size)).type_as(
                self.emb_gate_linear.weight
            )
            for step in range(self.kernel_size)
        ]

        we_cat = self.emb_gate_linear(embedding_hidden_state)


        all_hidden_buffer = []

        for i in range(num_steps):
            ##################################################################################################################################
            # extract feature
            # [bsz, seq_len, H] -> [bsz*seq_len, H]
            feature0 = self.attn_feat_extractor(initial_hidden_states, bool_mask, None)


            ##################################################################################################################################
            # update word node states
            # get states before
            initial_hidden_states_before = [
                (
                    self.get_hidden_states_before(
                        padding_list[step], initial_hidden_states, step + 1
                    )
                    * sequence_mask
                ).view(-1, hidden_size)
                for step in range(self.kernel_size)
            ]
            initial_hidden_states_before = self.sum_together(initial_hidden_states_before)

            initial_hidden_states_after = [
                (
                    self.get_hidden_states_after(
                        padding_list[step], initial_hidden_states, step + 1
                    )
                    * sequence_mask
                ).view(-1, hidden_size)
                for step in range(self.kernel_size)
            ]
            initial_hidden_states_after = self.sum_together(initial_hidden_states_after)

            ## get states after
            initial_cell_states_before = [
                (
                    self.get_hidden_states_before(padding_list[step], initial_cell_states, step + 1)
                    * sequence_mask
                ).view(-1, hidden_size)
                for step in range(self.kernel_size)
            ]
            initial_cell_states_before = self.sum_together(initial_cell_states_before)

            initial_cell_states_after = [
                (
                    self.get_hidden_states_after(padding_list[step], initial_cell_states, step + 1)
                    * sequence_mask
                ).view(-1, hidden_size)
                for step in range(self.kernel_size)
            ]
            initial_cell_states_after = self.sum_together(initial_cell_states_after)

            # reshape for matmul
            initial_hidden_states = initial_hidden_states.view(-1, hidden_size)
            initial_cell_states = initial_cell_states.view(-1, hidden_size)

            # concat before and after hidden states
            concat_before_after = torch.cat(
                [initial_hidden_states_before, initial_hidden_states_after], dim=1
            )


            # compute gate
            peep_cat = self.peep_gate_linear(initial_cell_states) # peephole connection

            cat_gate_value = (
                self.norm_gate(
                    torch.cat([initial_hidden_states, concat_before_after, feature0], dim=1)
                )
                + we_cat + peep_cat
            )
            c_fl, c_fr, c_fc, c_feat, c_i, c_w, c_o = torch.chunk(cat_gate_value, 7, dim=1)
            """
            c_fl, c_fr, c_fc: left, right and current cell gate
            c_feat: feature gate
            c_i: input gate
            c_w: write gate
            c_o: output gate
            """
            g_fl = torch.sigmoid(c_fl)
            g_fr = torch.sigmoid(c_fr)
            g_fc = torch.sigmoid(c_fc)
            g_feat = torch.sigmoid(c_feat)
            g_i = torch.sigmoid(c_i)
            g_w = torch.tanh(c_w)
            g_o = torch.sigmoid(c_o)

            gates_cat = torch.cat(
                [g_fl.unsqueeze(1), g_fr.unsqueeze(1), g_fc.unsqueeze(1), g_feat.unsqueeze(1), g_i.unsqueeze(1)], dim=1
            )
            gates_softmax = F.softmax(gates_cat, dim=1)
            g_fl, g_fr, g_fc, g_feat, g_i = torch.chunk(gates_softmax, 5, dim=1)

            g_fl, g_fr, g_fc, g_feat, g_i = g_fl.squeeze(1), g_fr.squeeze(1), g_fc.squeeze(1), g_feat.squeeze(1), g_i.squeeze(1)

            c_t_new =  initial_cell_states_before*g_fl + initial_cell_states_after*g_fr + initial_cell_states*g_fc + feature0*g_feat + g_w*g_i

            h_t_new = g_o * torch.tanh(c_t_new)

            # ffn blocks
            h_t_new = self.ffn_LN(h_t_new)
            res = h_t_new
            h_t_new = self.fc2(self.act_func(self.fc1(h_t_new)))
            h_t_new += res

            # update states
            initial_hidden_states = h_t_new.view(shape[0], src_len, hidden_size)
            initial_cell_states = c_t_new.view(shape[0], src_len, hidden_size)
            
            all_hidden_buffer.append(initial_hidden_states)

            # mask
            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask


            


        initial_hidden_states = F.dropout(
            initial_hidden_states, p=self.dropout, training=self.training
        )
        # initial_hidden_states = self.h_drop(self.mixture(hidden_buffer))
        # initial_cell_states = self.c_drop(initial_cell_states)


        all_mem_hidden = torch.stack(all_hidden_buffer, dim=0)       # [num_layers, bsz, src_len, H]
        # all_mem_hidden_inp = all_mem_hidden.transpose(1, 2).contiguous().view(num_steps * src_len, shape[0], hidden_size)       # [num_layers*src_len, bsz, H]
        
        # print('Time cost:', time.time() - time_s)
        # sys.exit(0)
        return initial_hidden_states, initial_cell_states, None, None, all_mem_hidden


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        # We do not need output projection as we have gates
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # init relative position embedding
        self.relpos_k = RelativePosition(self.head_dim)
        self.relpos_v = RelativePosition(self.head_dim)
        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        # if key_padding_mask is not None:
        #     print(key_padding_mask.size())
        #     xx
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]


        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                # print(key.size())
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            # print(k.size())
            # print(bsz * self.num_heads)
            # print("here")
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

            
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len



        attn_weights = torch.bmm(q, k.transpose(1, 2))
        ## relative position weights, 
        # q: bsz * self.num_heads, tgt_len, self.head_dim
        # r_k: tgt_len, src_len, self.head_dim
        q_len = tgt_len
        kv_len = src_len
        # if saved_state is not None:
        #     kv_len = saved_state.size()[2]
        # else:
        #     kv_len = src_len
        r_k = self.relpos_k(q_len, kv_len)
        # [tgt_len, bsz * self.num_heads, head_dim] x [tgt_len, self.head_dim, src_len]
        relpos_weights = torch.bmm(q.transpose(0,1), r_k.transpose(1,2)).transpose(0,1)
        # print(attn_weights.size())
        # print(relpos_weights.size())
        attn_weights += relpos_weights
        if saved_state is not None:
            print(tgt_len)
            # print(src_len)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)        # [bsz * self.num_heads, tgt_len, src_len]

        if before_softmax:
            return attn_weights, v

        attn_weights_float = fairseq_utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        ## relative position values, 
        # attn_probs: 
        # r_v: tgt_len, src_len, self.head_dim
        r_v = self.relpos_k(q_len, kv_len)
        # [tgt_len, bsz * self.num_heads, src_len] x [tgt_len, src_len, self.head_dim]
        relpos_attn = torch.bmm(attn_probs.transpose(0,1), r_v).transpose(0,1)
        attn += relpos_attn

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            # print(key_padding_mask.size())
            # xx
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)



    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position=16):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        # print(distance_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        # print(final_mat)
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings

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

        self.layer_norm = LayerNorm(self.embed_dim) 

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
        x = self.layer_norm(x)
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

        return x


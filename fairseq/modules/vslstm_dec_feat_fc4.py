# coding:utf-8
import math
import torch
from torch import Tensor, nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import sys
import time
from fairseq.modules import LayerNorm, TransformerEncoderLayer, LightweightConv, MultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.incremental_decoding_utils import with_incremental_state
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq import utils as fairseq_utils
from torch.nn import Parameter


class VanillaSLSTMFeatDecLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False):
        # current
        super(VanillaSLSTMFeatDecLayer, self).__init__()
        self.hidden_size = args.encoder_embed_dim
        self.dropout = args.dropout
        self.kernel_size = args.kernel_size

        self.norm_gate = nn.Linear(3 * self.hidden_size, 6 * self.hidden_size)
        self.emb_gate_linear = nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=False)
        self.temperature = args.temperature

        # feat lstm init
        # self.attn_feautre = TransformerEncoderLayer(args)
        # cross attention feature
        self.cross_attn_feautre = MultiheadAttention(
            self.hidden_size,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )
        # self.attn_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.fc2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.act_func = nn.ReLU(inplace=True)        
        self.ffn_LN = LayerNorm(self.hidden_size)


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

    def sum_together(self, ll):
        return sum(ll)
        
    def attn_feat_extractor(self, attn_module, q, k, v, key_padding_mask, incremental_state):
        q = q.transpose(0,1) #  -> [seq_len, bsz, H]
        k = k.transpose(0,1) #  -> [seq_len, bsz, H]
        v = v.transpose(0,1) #  -> [seq_len, bsz, H]

        key_padding_mask = torch.squeeze(key_padding_mask, dim=-1)

        # x = self.attn_feautre(x, encoder_padding_mask=key_padding_mask)
        out, attn = attn_module(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=False,
            need_head_weights=False,
        )
        out = out.transpose(0,1) # -> [bsz, seq_len, H]
        bsz, seq_len, hidden_size = out.size()

        out = torch.reshape(out, (seq_len*bsz,hidden_size))
        return out

    def forward(self, hidden_states, cell_states, we_cat, self_attn_mask, encoder_out, incremental_state, seq_length=None):

        if encoder_out is not None:
            encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
            encoder_out = encoder_out["encoder_out"][0]

        self_attn_mask = self_attn_mask

        # record shape of the batch
        # [bsz, seq_len, H]  
        shape = hidden_states.size()
        # randomly initialize the states
        initial_hidden_states = hidden_states
        initial_cell_states = cell_states

        batch_size, src_len, hidden_size = shape[0], shape[1], shape[2]
        padding_list = [
            self.create_padding_variable((batch_size, step + 1, hidden_size)).type_as(
                self.emb_gate_linear.weight
            )
            for step in range(self.kernel_size)
        ]



        ##################################################################################################################################
        # extract feature
        # [bsz, seq_len, H] -> [bsz*seq_len, H]
        # to replace with cross attention
        feature0 = self.attn_feat_extractor(self.cross_attn_feautre, initial_hidden_states, encoder_out, encoder_out, encoder_padding_mask, incremental_state)
        # feature0 = initial_hidden_states.view(-1, hidden_size)


        ##################################################################################################################################
        # update word node states
        # get states before
        initial_hidden_states_before = [
            (
                self.get_hidden_states_before(
                    padding_list[step], initial_hidden_states, step + 1
                )
            ).view(-1, hidden_size)
            for step in range(self.kernel_size)
        ]
        initial_hidden_states_before = self.sum_together(initial_hidden_states_before)
        ## get states before
        initial_cell_states_before = [
            (
                self.get_hidden_states_before(padding_list[step], initial_cell_states, step + 1)
            ).view(-1, hidden_size)
            for step in range(self.kernel_size)
        ]
        initial_cell_states_before = self.sum_together(initial_cell_states_before)

        # reshape for matmul
        initial_hidden_states = initial_hidden_states.view(-1, hidden_size)
        initial_cell_states = initial_cell_states.view(-1, hidden_size)

        # compute gate
        cat_gate_value = (
            self.norm_gate(
                torch.cat([initial_hidden_states, initial_hidden_states_before, feature0], dim=1)
            )
            + we_cat
        )
        c_fl, c_fc, c_feat, c_i, c_w, c_o = torch.chunk(cat_gate_value, 6, dim=1)
        """
        c_fl, c_fc: left and current cell gate
        c_feat: feature gate
        c_i: input gate
        c_w: write gate
        c_o: output gate
        """
        g_fl = torch.sigmoid(c_fl)
        g_fc = torch.sigmoid(c_fc)
        g_feat = torch.sigmoid(c_feat)
        g_i = torch.sigmoid(c_i)
        g_w = torch.tanh(c_w)
        g_o = torch.sigmoid(c_o)

        gates_cat = torch.cat(
            [g_fl.unsqueeze(1), g_fc.unsqueeze(1), g_feat.unsqueeze(1), g_i.unsqueeze(1)], dim=1
        )
        gates_softmax = F.softmax(gates_cat, dim=1)
        g_fl, g_fc, g_feat, g_i = torch.chunk(gates_softmax, 5, dim=1)

        g_fl, g_fc, g_feat, g_i = g_fl.squeeze(1), g_fc.squeeze(1), g_feat.squeeze(1), g_i.squeeze(1)

        c_t_new =  initial_cell_states_before*g_fl + initial_cell_states*g_fc + feature0*g_feat + g_w*g_i

        h_t_new = g_o * torch.tanh(c_t_new)

        # ffn blocks
        res = h_t_new
        h_t_new = self.ffn_LN(h_t_new)
        h_t_new = self.fc2(self.act_func(self.fc1(h_t_new)))
        h_t_new += res

        # update states
        initial_hidden_states = h_t_new.view(shape[0], src_len, hidden_size)
        initial_cell_states = c_t_new.view(shape[0], src_len, hidden_size)


        # mask
        initial_hidden_states = initial_hidden_states
        initial_cell_states = initial_cell_states
        initial_hidden_states = F.dropout(
            initial_hidden_states, p=self.dropout, training=self.training
        )

        return initial_hidden_states, initial_cell_states

    @torch.jit.unused
    def reorder_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    @torch.jit.unused
    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        return utils.get_incremental_state(self, incremental_state, "input_buffer")

    @torch.jit.unused
    def _set_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], new_buffer):
        return utils.set_incremental_state(
            self, incremental_state, "input_buffer", new_buffer
        )



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


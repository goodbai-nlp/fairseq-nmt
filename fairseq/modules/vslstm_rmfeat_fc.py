# coding:utf-8
import math
import torch
from torch import Tensor, nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import sys
import time
from fairseq.modules import LayerNorm, TransformerEncoderLayer
from fairseq.modules.quant_noise import quant_noise
from fairseq.incremental_decoding_utils import with_incremental_state
from typing import Dict, Optional, Tuple
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq import utils as fairseq_utils

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

        self.norm_gate = nn.Linear(3 * self.hidden_size, 7 * self.hidden_size)


        self.emb_gate_linear = nn.Linear(self.hidden_size, 7 * self.hidden_size, bias=False)

        self.peep_gate_linear = nn.Linear(self.hidden_size, 7 * self.hidden_size, bias=False)


        self.temperature = args.temperature
        # if self.config.HP_gpu:
        #     self.to(self.config.device)

        # reset slstm params
        self.reset_parameters()

        # feat lstm init
        # self.attn_feautre = TransformerEncoderLayer(args)
        # self.attn_feautre = MultiheadAttention(
        #     self.hidden_size,
        #     args.encoder_attention_heads,
        #     dropout=args.attention_dropout,
        #     self_attention=True,
        # )
        # self.attn_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
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
        sections = [self.hidden_size, 2 * self.hidden_size]
        self.Wxf1, self.Whf1 = torch.split(self.W_fl, sections, dim=0)  # [H, H]
        self.Wxf2, self.Whf2 = torch.split(self.W_fr, sections, dim=0)
        self.Wxf3, self.Whf3 = torch.split(self.W_fc, sections, dim=0)
        self.Wxf4, self.Whf4 = torch.split(self.W_fd, sections, dim=0)
        self.Wxi, self.Whi = torch.split(self.W_i, sections, dim=0)
        self.Wxt, self.Wht = torch.split(self.W_w, sections, dim=0)
        self.Wxo, self.Who = torch.split(self.W_o, sections, dim=0)
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
        nn.init.normal_(self.Wpf1, mean=0, std=0.1)

        nn.init.normal_(self.Wxf2, mean=0, std=0.1)
        nn.init.normal_(self.Whf2, mean=0, std=0.1)
        nn.init.normal_(self.Wif2, mean=0, std=0.1)
        nn.init.normal_(self.Wpf2, mean=0, std=0.1)

        nn.init.normal_(self.Wxf3, mean=0, std=0.1)
        nn.init.normal_(self.Whf3, mean=0, std=0.1)
        nn.init.normal_(self.Wif3, mean=0, std=0.1)
        nn.init.normal_(self.Wpf3, mean=0, std=0.1)

        nn.init.normal_(self.Wxf4, mean=0, std=0.1)
        nn.init.normal_(self.Whf4, mean=0, std=0.1)
        nn.init.normal_(self.Wif4, mean=0, std=0.1)
        nn.init.normal_(self.Wpf4, mean=0, std=0.1)

        nn.init.normal_(self.Wxi, mean=0, std=0.1)
        nn.init.normal_(self.Whi, mean=0, std=0.1)
        nn.init.normal_(self.Wii, mean=0, std=0.1)
        nn.init.normal_(self.Wpi, mean=0, std=0.1)

        nn.init.normal_(self.Wxt, mean=0, std=0.1)
        nn.init.normal_(self.Wht, mean=0, std=0.1)
        nn.init.normal_(self.Wit, mean=0, std=0.1)
        nn.init.normal_(self.Wpt, mean=0, std=0.1)

        nn.init.normal_(self.Wxo, mean=0, std=0.1)
        nn.init.normal_(self.Who, mean=0, std=0.1)
        nn.init.normal_(self.Wio, mean=0, std=0.1)
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

        x = self.attn_feautre(x, encoder_padding_mask=key_padding_mask)
        seq_len, bsz, hidden_size = x.size()
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



        ##################################################################################################################################
        ## extract features 
        # [bsz, seq_len, H] -> [bsz*seq_len, H]




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
                    torch.cat([initial_hidden_states, concat_before_after], dim=1)
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

            c_t_new =  initial_cell_states_before*g_fl + initial_cell_states_after*g_fr + initial_cell_states*g_fc + g_w*g_i

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

            ##################################################################################################################################

            


        initial_hidden_states = F.dropout(
            initial_hidden_states, p=self.dropout, training=self.training
        )
        # initial_hidden_states = self.h_drop(self.mixture(hidden_buffer))
        # initial_cell_states = self.c_drop(initial_cell_states)


        all_mem_hidden = torch.stack(all_hidden_buffer, dim=0)       # [num_layers, bsz, src_len, H]
        all_mem_hidden_inp = all_mem_hidden.transpose(1, 2).contiguous().view(num_steps * src_len, shape[0], hidden_size)       # [num_layers*src_len, bsz, H]
        
        # print('Time cost:', time.time() - time_s)
        # sys.exit(0)
        return initial_hidden_states, initial_cell_states, None, None, all_mem_hidden



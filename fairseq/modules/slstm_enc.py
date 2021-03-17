# coding:utf-8
import torch
from torch import nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
import sys
import time

from fairseq.modules import LayerNorm


class SLSTMEncLayer(nn.Module):
    def __init__(self, args):
        # current
        super(SLSTMEncLayer, self).__init__()
        self.hidden_size = args.encoder_embed_dim
        self.dropout = args.dropout
        self.kernel_size = args.kernel_size
        # self.l_base = args.base_layer
        # self.l_std = args.std_layer
        # self.s_alpha = args.sigmoid_alpha

        self.norm_gate = nn.Linear(4 * self.hidden_size, 8 * self.hidden_size)
        self.dummy_gate = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)

        self.emb_gate_linear = nn.Linear(self.hidden_size, 8 * self.hidden_size, bias=False)

        self.dummy_fgate = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.emb_trans = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.h_trans = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.h_reset_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.act_func = nn.ReLU(inplace=True)

        # self.h_drop = nn.Dropout(self.dropout)
        # self.c_drop = nn.Dropout(self.dropout)
        # self.g_drop = nn.Dropout(self.dropout)

        # self.input_norm = LayerNorm(hidden_size)
        self.i_norm = LayerNorm(self.hidden_size)
        self.o_norm = LayerNorm(self.hidden_size)
        self.l_norm = LayerNorm(self.hidden_size)
        self.f1_norm = LayerNorm(self.hidden_size)
        self.f2_norm = LayerNorm(self.hidden_size)
        self.f3_norm = LayerNorm(self.hidden_size)
        self.f4_norm = LayerNorm(self.hidden_size)

        self.gd_norm = LayerNorm(self.hidden_size)
        self.go_norm = LayerNorm(self.hidden_size)
        self.gf_norm = LayerNorm(self.hidden_size)
        self.h_LN = LayerNorm(self.hidden_size)

        # self.c_norm = LayerNorm(self.hidden_size)
        # self.gc_norm = LayerNorm(self.hidden_size)
        # self.gh_norm = LayerNorm(self.hidden_size)
        self.temperature = args.temperature
        # if self.config.HP_gpu:
        #     self.to(self.config.device)

        self.reset_parameters()

    def reset_parameters(self):
        self.norm_gates_W = self.norm_gate.weight.t()
        self.norm_gates_b = self.norm_gate.bias
        (
            self.W_fl,
            self.W_fr,
            self.W_fc,
            self.W_fd,
            self.W_i,
            self.W_e,
            self.W_w,
            self.W_o,
        ) = torch.chunk(
            self.norm_gates_W, 8, dim=1
        )  # [4*H, H]
        sections = [self.hidden_size, 2 * self.hidden_size, self.hidden_size]
        self.Wxf1, self.Whf1, self.Wdf1 = torch.split(self.W_fl, sections, dim=0)  # [H, H]
        self.Wxf2, self.Whf2, self.Wdf2 = torch.split(self.W_fr, sections, dim=0)
        self.Wxf3, self.Whf3, self.Wdf3 = torch.split(self.W_fc, sections, dim=0)
        self.Wxf4, self.Whf4, self.Wdf4 = torch.split(self.W_fd, sections, dim=0)
        self.Wxi, self.Whi, self.Wdi = torch.split(self.W_i, sections, dim=0)
        self.Wxt, self.Wht, self.Wdt = torch.split(self.W_w, sections, dim=0)
        self.Wxl, self.Whl, self.Wdl = torch.split(self.W_e, sections, dim=0)
        self.Wxo, self.Who, self.Wdo = torch.split(self.W_o, sections, dim=0)
        (
            self.b_fl,
            self.b_fr,
            self.b_fc,
            self.b_fd,
            self.bi,
            self.be,
            self.bw,
            self.bo,
        ) = torch.chunk(self.norm_gates_b, 8, dim=0)
        self.emb_gates_w = self.emb_gate_linear.weight.t()
        (
            self.Wif1,
            self.Wif2,
            self.Wif3,
            self.Wif4,
            self.Wii,
            self.Wil,
            self.Wit,
            self.Wio,
        ) = torch.chunk(self.emb_gates_w, 8, dim=1)

        self.dummy_gates_W = self.dummy_gate.weight.t()
        self.dummy_gates_b = self.dummy_gate.bias
        self.W_fd_d, self.W_od = torch.chunk(self.dummy_gates_W, 2, dim=1)  # [2*H, H]
        self.gated_Wxd, self.gated_Whd = torch.chunk(self.W_fd_d, 2, dim=0)
        self.gated_Wxo, self.gated_Who = torch.chunk(self.W_od, 2, dim=0)
        self.gated_bd, self.gated_bo = torch.chunk(self.dummy_gates_b, 2, dim=0)

        self.dummy_fgates_W = self.dummy_fgate.weight.t()
        self.dummy_fgates_b = self.dummy_fgate.bias
        self.gated_Wxf, self.gated_Whf = torch.chunk(self.dummy_fgates_W, 2, dim=0)
        self.gated_bf = self.dummy_fgates_b
        self.We = self.emb_trans.weight.t()

        self.initializer = nn.init.normal_
        # self.initializer = nn.init.xavier_normal_
        # self.initializer = nn.init.orthogonal_

        self.initializer(self.Wxf1, mean=0, std=0.1)
        nn.init.normal_(self.Whf1, mean=0, std=0.1)
        nn.init.normal_(self.Wif1, mean=0, std=0.1)
        nn.init.normal_(self.Wdf1, mean=0, std=0.1)

        nn.init.normal_(self.Wxf2, mean=0, std=0.1)
        nn.init.normal_(self.Whf2, mean=0, std=0.1)
        nn.init.normal_(self.Wif2, mean=0, std=0.1)
        nn.init.normal_(self.Wdf2, mean=0, std=0.1)

        nn.init.normal_(self.Wxf3, mean=0, std=0.1)
        nn.init.normal_(self.Whf3, mean=0, std=0.1)
        nn.init.normal_(self.Wif3, mean=0, std=0.1)
        nn.init.normal_(self.Wdf3, mean=0, std=0.1)

        nn.init.normal_(self.Wxf4, mean=0, std=0.1)
        nn.init.normal_(self.Whf4, mean=0, std=0.1)
        nn.init.normal_(self.Wif4, mean=0, std=0.1)
        nn.init.normal_(self.Wdf4, mean=0, std=0.1)

        nn.init.normal_(self.Wxi, mean=0, std=0.1)
        nn.init.normal_(self.Whi, mean=0, std=0.1)
        nn.init.normal_(self.Wii, mean=0, std=0.1)
        nn.init.normal_(self.Wdi, mean=0, std=0.1)

        nn.init.normal_(self.Wxt, mean=0, std=0.1)
        nn.init.normal_(self.Wht, mean=0, std=0.1)
        nn.init.normal_(self.Wit, mean=0, std=0.1)
        nn.init.normal_(self.Wdt, mean=0, std=0.1)

        nn.init.normal_(self.Wxl, mean=0, std=0.1)
        nn.init.normal_(self.Whl, mean=0, std=0.1)
        nn.init.normal_(self.Wil, mean=0, std=0.1)
        nn.init.normal_(self.Wdl, mean=0, std=0.1)

        nn.init.normal_(self.We, mean=0, std=0.1)

        nn.init.normal_(self.Wxo, mean=0, std=0.1)
        nn.init.normal_(self.Who, mean=0, std=0.1)
        nn.init.normal_(self.Wio, mean=0, std=0.1)
        nn.init.normal_(self.Wdo, mean=0, std=0.1)

        nn.init.normal_(self.bi, mean=0, std=0.1)
        nn.init.normal_(self.bw, mean=0, std=0.1)
        nn.init.normal_(self.be, mean=0, std=0.1)
        nn.init.normal_(self.bo, mean=0, std=0.1)

        nn.init.normal_(self.b_fl, mean=0, std=0.1)
        nn.init.normal_(self.b_fr, mean=0, std=0.1)
        nn.init.normal_(self.b_fc, mean=0, std=0.1)
        nn.init.normal_(self.b_fd, mean=0, std=0.1)

        # stdv = 1. / math.sqrt(self.bf1.shape[0])
        # torch.nn.init.uniform_(self.bf1, a=-stdv, b=stdv)
        # torch.nn.init.uniform_(self.bf2, a=-stdv, b=stdv)
        # torch.nn.init.uniform_(self.bf3, a=-stdv, b=stdv)
        # torch.nn.init.uniform_(self.bf4, a=-stdv, b=stdv)

        # forget gate initialize to 0
        # nn.init.constant_(self.bf1, 0.)
        # nn.init.constant_(self.bf2, 0.)
        # nn.init.constant_(self.bf3, 0.)
        # nn.init.constant_(self.bf4, 0.)

        nn.init.normal_(self.gated_Wxd, mean=0, std=0.1)
        nn.init.normal_(self.gated_Whd, mean=0, std=0.1)
        nn.init.normal_(self.gated_Wxo, mean=0, std=0.1)
        nn.init.normal_(self.gated_Who, mean=0, std=0.1)
        nn.init.normal_(self.gated_Wxf, mean=0, std=0.1)
        nn.init.normal_(self.gated_Whf, mean=0, std=0.1)

        nn.init.normal_(self.gated_bd, mean=0, std=0.1)

        nn.init.normal_(self.gated_bo, mean=0, std=0.1)

        nn.init.normal_(self.gated_bf, mean=0, std=0.1)

        # nn.init.constant_(self.gated_bf, 0.0)
        # stdv = 1. / math.sqrt(self.gated_bf.shape[0])
        # torch.nn.init.uniform_(self.gated_bf, a=-stdv, b=stdv)

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

    # def my_sigmoid(self, x):
    #     return 1 / (1 + np.exp(-self.s_alpha * x))

    def forward(self, word_inputs, mask, num_layers, seq_length=None):
        # print('using fast3!!!!!!')
        # filters for attention
        # num_steps = 6 + np.around(4*(torch.sigmoid(mean_len-24)-0.5))
        # mean_len = torch.mean(seq_length.float()).cpu().numpy()
        # num_steps = int(
        #     self.l_base + np.around(self.l_std * (self.my_sigmoid(mean_len - 24) - 0.5))
        # )

        # time_s = time.time()
        # self.update_cat_param()

        # print('update cost:', time.time() - time_s)
        # time_s = time.time()
        num_steps = num_layers

        # [bsz, src_len]
        mask_softmax_score = -mask.float() * 1e25
        mask_softmax_score_expanded = torch.unsqueeze(mask_softmax_score, dim=2).type_as(
            self.emb_trans.weight
        )  # 10, 40, 1
        # print("[tlog] mask_softmax_expanded: " + str(mask_softmax_score_expanded))

        # filter invalid steps
        sequence_mask = torch.unsqueeze(
            1 - mask.type_as(self.emb_trans.weight), dim=2
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

        # inital dummy node states
        dummynode_hidden_states = (
            torch.sum(initial_hidden_states, dim=1) / sequence_lengths
        )  # [bsz, H]
        # dummynode_hidden_states = self.i_drop(dummynode_hidden_states)
        dummynode_cell_states = torch.sum(initial_cell_states, dim=1) / sequence_lengths  # [bsz, H]
        # print("[tlog] emb_hidden_state", embedding_hidden_state.size())

        # hidden_size = self.hidden_size
        batch_size, src_len, hidden_size = shape[0], shape[1], shape[2]
        padding_list = [
            self.create_padding_variable((batch_size, step + 1, hidden_size)).type_as(
                self.emb_trans.weight
            )
            for step in range(self.kernel_size)
        ]

        # we_cat = torch.matmul(embedding_hidden_state, self.emb_gates_w)
        we_cat = self.emb_gate_linear(embedding_hidden_state)
        # emb_fl, emb_fr, emb_fc, emb_fd, emb_i, emb_e, emb_w, emb_o = torch.split(we_cat, split_size_or_sections=self.hidden_size, dim=1)
        # emb_fl, emb_fr, emb_fc, emb_fd, emb_i, emb_e, emb_w, emb_o = torch.chunk(we_cat, 8, dim=1)

        # emb_trans = self.emb_trans(embedding_hidden_state)
        emb_trans = torch.tanh(self.emb_trans(embedding_hidden_state))

        global_hidden_buffer, global_cell_buffer, all_hidden_buffer = [], [], []

        for i in range(num_steps):
            # print("[tlog] layers: " + str(i))
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

            # get states after
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

            # copy dummy node states

            transformed_dummynode_cell_states = (
                torch.unsqueeze(dummynode_cell_states, dim=1)
                .repeat(1, src_len, 1)
                .view(-1, hidden_size)
            )

            transformed_dummynode_hidden_states = torch.unsqueeze(
                dummynode_hidden_states, dim=1
            ).repeat(1, src_len, 1)

            transformed_dummynode_hidden_states = (
                transformed_dummynode_hidden_states * sequence_mask
            ).view(
                -1, hidden_size
            )  # add 2019-03-12

            dummy_unsqueezed = (
                dummynode_hidden_states.unsqueeze(dim=1).repeat(1, src_len, 1).view(-1, hidden_size)
            )  # [bsz*src_len, H]
            cat_gate_value = (
                self.norm_gate(
                    torch.cat([initial_hidden_states, concat_before_after, dummy_unsqueezed], dim=1)
                )
                + we_cat
            )
            c_fl, c_fr, c_fc, c_fd, c_i, c_e, c_w, c_o = torch.chunk(cat_gate_value, 8, dim=1)

            g_fl = self.f1_norm(c_fl)
            g_fr = self.f2_norm(c_fr)
            g_fc = self.f3_norm(c_fc)
            g_fd = torch.sigmoid(self.f4_norm(c_fd))
            g_i = torch.sigmoid(self.i_norm(c_i))
            g_w = torch.tanh(c_w)
            g_e = torch.sigmoid(self.l_norm(c_e))
            g_o = torch.sigmoid(self.o_norm(c_o))

            g_w = g_w + g_e * emb_trans

            first_three_gates_new = torch.cat(
                [g_fl.unsqueeze(1), g_fr.unsqueeze(1), g_fc.unsqueeze(1)], dim=1
            )
            first_three_gates_new = F.softmax(first_three_gates_new, dim=1)
            g_fl, g_fr, g_fc = torch.chunk(first_three_gates_new, 3, dim=1)
            g_fl, g_fr, g_fc = g_fl.squeeze(1), g_fr.squeeze(1), g_fc.squeeze(1)

            local_c_t_new = (
                (initial_cell_states_before * g_fl)
                + (initial_cell_states_after * g_fr)
                + (initial_cell_states * g_fc)
            )

            local_and_global_c_t_new = local_c_t_new * (1.0 - g_fd) + (
                transformed_dummynode_cell_states * g_fd
            )

            c_t_new = g_w * g_i + local_and_global_c_t_new * (1.0 - g_i)

            # h_t = o_t * torch.tanh(self.c_norm(c_t)) #+ (1.0 - o_t) * embedding_hidden_state

            # h_t = o_t * torch.tanh(c_t) #+ (1.0 - o_t) * embedding_hidden_state
            # h_t = o_t * c_t +  reshaped_hidden_output # 92.75
            # c_t =  c_t +  reshaped_hidden_output      # 92.75
            # h_t = o_t * c_t
            # ## h_t = o_t * (c_t +  reshaped_hidden_output)

            reshaped_hidden_output = initial_hidden_states.view(-1, hidden_size)

            # positionwise_ff = self.fc2(self.act_func(self.fc1(c_t_new)))
            # h_t_new = g_o * torch.tanh(positionwise_ff + c_t_new)
            h_t_new = g_o * torch.tanh(c_t_new)
            h_t_new = self.h_LN(h_t_new + reshaped_hidden_output)

            # FFN layer here
            # print("new_version"*100)
            res = h_t_new
            h_t_new = self.fc2(self.act_func(self.fc1(h_t_new)))
            h_t_new = res + h_t_new

            # update states
            initial_hidden_states = h_t_new.view(shape[0], src_len, hidden_size)
            initial_cell_states = c_t_new.view(shape[0], src_len, hidden_size)
            
            all_hidden_buffer.append(initial_hidden_states)

            # mask
            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask
            ##################################################################################################################################
            # update dummy node states
            # average states
            combined_word_hidden_state = torch.sum(initial_hidden_states, dim=1) / sequence_lengths
            # print("[tlog] combined_word_hidden_state: " + str(combined_word_hidden_state))
            reshaped_hidden_output = initial_hidden_states.view(-1, hidden_size)
            # print("dummy_hidden", dummynode_hidden_states.size())       # [bsz, H]
            # print("combined_word_hidden_state", combined_word_hidden_state.size())  # [bsz, H]

            cat_gate_dummy = self.dummy_gate(
                torch.cat([dummynode_hidden_states, combined_word_hidden_state], dim=1)
            )
            dummy_fd_v, dummy_o_v = torch.chunk(cat_gate_dummy, 2, dim=1)

            gated_d_t = torch.sigmoid(self.gd_norm(dummy_fd_v))
            gated_o_t = torch.sigmoid(self.go_norm(dummy_o_v))

            gated_f_t_v = self.dummy_fgate(
                torch.cat([transformed_dummynode_hidden_states, reshaped_hidden_output], dim=1)
            )
            gated_f_t = torch.sigmoid(self.gf_norm(gated_f_t_v))

            reshaped_gated_f_t = (
                gated_f_t.view(shape[0], src_len, hidden_size) + mask_softmax_score_expanded
            )

            gated_softmax_scores = F.softmax(
                torch.cat([reshaped_gated_f_t, torch.unsqueeze(gated_d_t, dim=1)], dim=1)
                / self.temperature,
                dim=1,
            )

            # print("[tlog] gated_softmax_scores: " + str(gated_softmax_scores))
            # gated_softmax_scores = F.dropout(gated_softmax_scores.permute(0, 2, 1), p=self.dropout, training=self.training).permute(0,2,1)
            # print("[tlog] gated_softmax_scores: " + str(gated_softmax_scores))

            # self.debug = gated_softmax_scores
            # split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:, :src_len, :]
            new_gated_d_t = gated_softmax_scores[:, src_len:, :]       # [bsz, 1, H]
            # print("[tlog] new_reshaped_gated_f_t: " + str(new_reshaped_gated_f_t))
            # print("[tlog] new_gated_d_t: " + str(new_gated_d_t))

            # new dummy states
            dummy_c_t = (
                torch.sum(new_reshaped_gated_f_t * initial_cell_states, dim=1)
                + torch.squeeze(new_gated_d_t, dim=1) * dummynode_cell_states
            )

            dummy_h_t = gated_o_t * torch.tanh(dummy_c_t)
            # dummy_h_t = self.gh_norm(gated_o_t * torch.tanh(dummy_c_t) + dummynode_hidden_states)
            # sys.exit(0)
            ##########################################################################################################################################################
            ##################################################################################################################################

            dummynode_hidden_states = dummy_h_t
            dummynode_cell_states = dummy_c_t

            # initial_hidden_states = self.locked_dropout(initial_hidden_states, self.config.HP_dropout)
            # dummynode_hidden_states = self.i_drop(dummynode_hidden_states)
            # hidden_buffer.append(initial_hidden_states)

            global_hidden_buffer.append(dummynode_hidden_states)
            global_cell_buffer.append(dummynode_cell_states)

        initial_hidden_states = F.dropout(
            initial_hidden_states, p=self.dropout, training=self.training
        )
        # initial_hidden_states = self.h_drop(self.mixture(hidden_buffer))
        # initial_cell_states = self.c_drop(initial_cell_states)

        all_dummy_hidden = torch.stack(global_hidden_buffer, dim=0)  # [num_layers, bsz, H]
        all_dummy_cell = torch.stack(global_cell_buffer, dim=0)  # [num_layers, bsz, H]
        all_mem_hidden = torch.stack(all_hidden_buffer, dim=0)       # [num_layers, bsz, src_len, H]
        all_mem_hidden_inp = all_mem_hidden.transpose(1, 2).contiguous().view(num_steps * src_len, shape[0], hidden_size)       # [num_layers*src_len, bsz, H]
        
        # print('Time cost:', time.time() - time_s)
        # sys.exit(0)
        return initial_hidden_states, initial_cell_states, all_dummy_hidden, all_dummy_cell, all_mem_hidden
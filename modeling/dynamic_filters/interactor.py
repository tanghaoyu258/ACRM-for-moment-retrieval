import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell, Linear, Parameter


class InteractorwoLSTM(nn.Module):
    def __init__(self, hidden_size_textual: int, hidden_size_visual: int,
                 hidden_size_ilstm: int):
        """
        :param input_size:
        :param hidden_size:
        """
        super(InteractorwoLSTM, self).__init__()

        # represented by W_S, W_R, W_V with bias b
        self.projection_S = Linear(hidden_size_textual, hidden_size_ilstm, bias=True)
        self.projection_V = Linear(hidden_size_visual, hidden_size_ilstm, bias=True)
        #self.projection_R = Linear(hidden_size_ilstm, hidden_size_ilstm, bias=True)

        # parameter w with bias c
        self.projection_w = Linear(hidden_size_ilstm, 1, bias=True)

        self.hidden_size_textual = hidden_size_textual
        self.hidden_size_visual = hidden_size_visual
        self.hidden_size_ilstm = hidden_size_ilstm

        # self.iLSTM = LSTMCell(input_size=hidden_size_textual,
        #                       hidden_size=hidden_size_ilstm)

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def forward(self, h_s: torch.Tensor, h_v: torch.Tensor,  lengths=None,):
        """
        :param h_v: with shape (n_batch, T, hidden_size_visual)
        :param h_s: with shape (n_batch, N, hidden_size_textual)
        :return: outputs of the iLSTM with shape (n_batch, T, hidden_size_ilstm)
        """
        n_batch, T, N = h_v.shape[0], h_v.shape[1], h_s.shape[1]

        # h_r_{t-1} in the paper
        # h_r_prev = torch.zeros([n_batch, self.hidden_size_ilstm], device=self.device)
        # c_r_prev = torch.zeros([n_batch, self.hidden_size_ilstm], device=self.device)
        token_mask = self.get_mask_from_sequence_lengths(lengths,N) #(n_batch, N)
        outputs = []
        attention_weights = []
        for t in range(T):
            beta_t = self.projection_w(torch.tanh(self.projection_S(h_s) +
                                                  self.projection_V(h_v[:, t, :]).unsqueeze(dim=1))
                                       ).squeeze(dim=2)  # shape (n_batch, N)

            # alpha_t = torch.softmax(beta_t, dim=1)  # shape: (n_batch, N)
            alpha_t = self.masked_softmax(beta_t,token_mask,dim=1)
            # computing H_t_s with shape (n_batch, hidden_size_textual)
            H_t_s = torch.bmm(h_s.permute(0, 2, 1), alpha_t.unsqueeze(dim=2)).squeeze(dim=2)

            #r_t = torch.cat([h_v[:, t, :], H_t_s], dim=1)  # shape (n_batch, hidden_size_textual+hidden_size_visual)
            # r_t = h_v[:, t, :] - H_t_s
            # computing h_r_new and c_r_new with shape (n_batch, hidden_size_ilstm)
            # h_r_new, c_r_new = self.iLSTM(r_t, (h_r_prev, c_r_prev))
            outputs.append(H_t_s.unsqueeze(dim=1))
            # h_r_prev, c_r_prev = h_r_new, c_r_new
            # attention_weights.append(alpha_t.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)

    @property
    def device(self) -> torch.device:
        """
        Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.projection_S.weight.device
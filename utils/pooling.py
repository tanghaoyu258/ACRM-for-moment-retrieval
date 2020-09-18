import torch
from utils.rnns import (mean_pooling,
                    max_pooling,
                    gather_last)
import torch.nn as nn
from torch.nn import LSTM, LSTMCell, Linear, Parameter

class MeanPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()


    def forward(self, batch_hidden_states, video_fea, lengths, **kwargs):
        return mean_pooling(batch_hidden_states, lengths)


class MaxPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, batch_hidden_states,video_fea , lengths, **kwargs):
        return max_pooling(batch_hidden_states, lengths)


class GatherLastLayer(torch.nn.Module):

    def __init__(self, bidirectional=True):
        super(GatherLastLayer, self).__init__()
        self.bidirectional = bidirectional

    def forward(self, batch_hidden_states, video_fea , lengths, **kwargs):
        return gather_last(batch_hidden_states, lengths,
                           bidirectional=self.bidirectional)


class GatherFirstLayer(torch.nn.Module):

    def __init__(self):
        super(GatherFirstLayer, self).__init__()

    def forward(self, batch_hidden_states,video_fea , lengths, **kwargs):
        return batch_hidden_states[:,0,:]




import torch
import torch.nn as nn
from utils.rnns import feed_forward_rnn

class LSTM_VIDEO(nn.Module):

    def __init__(self, cfg):
        super(LSTM_VIDEO, self).__init__()

        self.input_size   = cfg.DYNAMIC_FILTER.LSTM_VIDEO.INPUT_SIZE
        self.num_layers   = cfg.DYNAMIC_FILTER.LSTM_VIDEO.NUM_LAYERS
        self.hidden_size  = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE
        self.bias         = cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIAS
        self.dropout      = cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT
        self.bidirectional= cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL
        self.batch_first  = cfg.DYNAMIC_FILTER.LSTM_VIDEO.BATCH_FIRST
        if cfg.DATASETS.TRAIN == 'anet_cap_train' and cfg.FEATURE_TYPE == 'c3d':
            self.input_size = 500
        if (cfg.DATASETS.TRAIN == "tacos_train" or cfg.DATASETS.TRAIN == "charades_sta_train") and cfg.FEATURE_TYPE == 'c3d':
            self.input_size = 4096
        self.lstm = nn.LSTM(input_size   = self.input_size,
                            hidden_size  = self.hidden_size,
                            num_layers   = self.num_layers,
                            bias         = self.bias,
                            dropout      = self.dropout,
                            bidirectional= self.bidirectional,
                            batch_first = self.batch_first)

    def forward(self, sequences, lengths, masks=None):
        if lengths is None:
            raise "ERROR in this tail you need lengths of sequences."
        
        return feed_forward_rnn(self.lstm,
                                sequences,
                                lengths=lengths)
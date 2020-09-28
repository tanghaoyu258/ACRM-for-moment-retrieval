import torch
import numpy as np
from torch import nn
import modeling.dynamic_filters as DF
import utils.pooling as POOLING

# from interactor import InteractorwoLSTM
class DynamicFilter(nn.Module):
    def __init__(self, cfg):
        super(DynamicFilter, self).__init__()
        self.cfg = cfg

        factory = getattr(DF, cfg.DYNAMIC_FILTER.TAIL_MODEL)
        self.tail_df = factory(cfg)

        factory = getattr(POOLING, cfg.DYNAMIC_FILTER.POOLING)
        self.pooling_layer = factory()

        factory = getattr(DF, cfg.DYNAMIC_FILTER.HEAD_MODEL)
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths=None, video_fea=None):
        output, _ = self.tail_df(sequences, lengths)
        output = self.pooling_layer(output, video_fea=None, lengths=lengths)
        output = self.head_df(output)
        return output, lengths 

class ACRM_query(nn.Module):
    def __init__(self, cfg):
        super(ACRM_query, self).__init__()
        self.cfg = cfg
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        factory = getattr(DF, cfg.ACRM_QUERY.TAIL_MODEL)
        self.tail_df = factory(cfg)

        factory = getattr(POOLING, cfg.ACRM_QUERY.POOLING)
        self.pooling_layer = factory()

        factory = getattr(DF, cfg.ACRM_QUERY.HEAD_MODEL)
        self.head_df = factory(cfg)
        self.query_hidden =cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL))
        self.video_hidden = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL))
        if cfg.ACRM_CLASSIFICATION.FUSION == 'CROSS_COS' or cfg.ACRM_CLASSIFICATION.FUSION == 'CROSS_SUB':
            self.pooling_layer = DF.InteractorwoLSTM(self.query_hidden, self.video_hidden, self.query_hidden)

    def forward(self, sequences, lengths=None, video_fea=None):
        output, _ = self.tail_df(sequences, lengths)
        output = self.dropout_layer(output)
        output = self.pooling_layer(output, video_fea, lengths)
        output = self.head_df(output)
        return output, lengths

class ACRM_video(nn.Module):
    def __init__(self, cfg):
        super(ACRM_video, self).__init__()
        self.cfg = cfg
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        factory = getattr(DF, cfg.ACRM_VIDEO.TAIL_MODEL)
        self.tail_df = factory(cfg)

        # factory = getattr(POOLING, cfg.ACRM_QUERY.POOLING)
        # self.pooling_layer = factory()

        factory = getattr(DF, cfg.ACRM_VIDEO.HEAD_MODEL)
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths, masks = None):
        output, _ = self.tail_df(sequences, lengths, masks)
        output = self.dropout_layer(output)
        # output = self.pooling_layer(output, lengths)
        output = self.head_df(output)
        return output

class GCN_map(nn.Module):
    def __init__(self, cfg):
        super(GCN_map, self).__init__()
        self.cfg = cfg
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        factory = getattr(DF, cfg.ACRM_VIDEO.TAIL_MODEL)
        self.tail_df = factory(cfg)

        # factory = getattr(POOLING, cfg.ACRM_QUERY.POOLING)
        # self.pooling_layer = factory()

        factory = getattr(DF, cfg.ACRM_VIDEO.HEAD_MODEL)
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths, masks = None):
        output, _ = self.tail_df(sequences, lengths, masks)
        output = self.dropout_layer(output)
        # output = self.pooling_layer(output, lengths)
        output = self.head_df(output)
        return output
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
# from modeling.dynamic_filters.multiheadatt import TransformerBlock
from modeling.dynamic_filters.build import DynamicFilter,ACRM_query,ACRM_video
from utils import loss as L
from utils.rnns import feed_forward_rnn
import utils.pooling as POOLING

class Localization_ACRM(nn.Module):
    def __init__(self, cfg):
        super(Localization_ACRM, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE_TRAIN
        self.model_df  = ACRM_query(cfg)
        # if cfg.ACRM_VIDEO.TAIL_MODEL == "LSTM":
        #     self.model_video_GRU = nn.LSTM(input_size   = cfg.DYNAMIC_FILTER.LSTM_VIDEO.INPUT_SIZE,
        #                                     num_layers   = cfg.DYNAMIC_FILTER.LSTM_VIDEO.NUM_LAYERS,
        #                                     hidden_size  = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE,
        #                                     bias         = cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIAS,
        #                                     dropout      = cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT,
        #                                     bidirectional= cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL,
        #                                     batch_first  = cfg.DYNAMIC_FILTER.LSTM_VIDEO.BATCH_FIRST)
        
        self.model_video_GRU = ACRM_video(cfg) 
        # self.reduction  = nn.Linear(cfg.REDUCTION.INPUT_SIZE, cfg.REDUCTION.OUTPUT_SIZE)
        self.multimodal_fc1 = nn.Linear(512*2, 1)
        self.multimodal_fc2 = nn.Linear(512, 1)

        self.is_use_rnn_loc = cfg.ACRM_CLASSIFICATION.USED
        self.rnn_localization = nn.LSTM(input_size   = cfg.ACRM_CLASSIFICATION.INPUT_SIZE,
                                        hidden_size  = cfg.ACRM_CLASSIFICATION.INPUT_SIZE_RNN,
                                        num_layers   = cfg.LOCALIZATION.ACRM_NUM_LAYERS,
                                        bias         = cfg.LOCALIZATION.BIAS,
                                        dropout      = cfg.LOCALIZATION.DROPOUT,
                                        bidirectional= cfg.LOCALIZATION.BIDIRECTIONAL,
                                        batch_first = cfg.LOCALIZATION.BATCH_FIRST)

        if cfg.ACRM_CLASSIFICATION.FUSION == 'CAT':
            cfg.ACRM_CLASSIFICATION.INPUT_SIZE = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL)) \
                + cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL))

        else:
            assert cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL)) == \
                cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL))
            cfg.ACRM_CLASSIFICATION.INPUT_SIZE = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL))
        if cfg.ACRM_CLASSIFICATION.USED == True:
            
            cfg.ACRM_CLASSIFICATION.INPUT_SIZE = cfg.ACRM_CLASSIFICATION.INPUT_SIZE_RNN * (1 + int(cfg.LOCALIZATION.BIDIRECTIONAL))
        
        self.pooling = POOLING.MeanPoolingLayer()
        self.starting = nn.Sequential(
            nn.Linear(cfg.ACRM_CLASSIFICATION.INPUT_SIZE, cfg.ACRM_CLASSIFICATION.HIDDEN_SIZE),
            nn.Tanh(),
            # nn.Dropout(cfg.LOCALIZATION.ACRM_DROPOUT),
            nn.Linear(cfg.ACRM_CLASSIFICATION.HIDDEN_SIZE, cfg.ACRM_CLASSIFICATION.OUTPUT_SIZE))
        self.ending = nn.Sequential(
            nn.Linear(cfg.ACRM_CLASSIFICATION.INPUT_SIZE, cfg.ACRM_CLASSIFICATION.HIDDEN_SIZE),
            nn.Tanh(),
            # nn.Dropout(cfg.LOCALIZATION.ACRM_DROPOUT),
            nn.Linear(cfg.ACRM_CLASSIFICATION.HIDDEN_SIZE, cfg.ACRM_CLASSIFICATION.OUTPUT_SIZE))            
        self.intering = nn.Sequential(
            nn.Linear(cfg.ACRM_CLASSIFICATION.INPUT_SIZE, cfg.ACRM_CLASSIFICATION.HIDDEN_SIZE),
            nn.Tanh(),
            # nn.Dropout(cfg.LOCALIZATION.ACRM_DROPOUT),
            nn.Linear(cfg.ACRM_CLASSIFICATION.HIDDEN_SIZE, cfg.ACRM_CLASSIFICATION.OUTPUT_SIZE))
        
        # self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)

        # self.starting = nn.Linear(cfg.CLASSIFICATION.INPUT_SIZE, cfg.CLASSIFICATION.OUTPUT_SIZE)
        # self.ending = nn.Linear(cfg.CLASSIFICATION.INPUT_SIZE, cfg.CLASSIFICATION.OUTPUT_SIZE)

    def attention(self, videoFeat, filter, lengths):
        pred_local = torch.bmm(videoFeat, filter.unsqueeze(2)).squeeze()
        return pred_local

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def feature_l2_normalize(self, data_tensor):
        mu = torch.norm(data_tensor,dim=-1, keepdim=True)
        data_tensor = data_tensor/mu
        return data_tensor

    def feature_gauss_normalize(self, data_tensor):
        mu = torch.mean(data_tensor,dim=-1,keepdim=True)
        std_value = torch.std(data_tensor,dim=-1,keepdim=True)
        return (data_tensor - mu)/std_value 

    def fusion_layer(self , filter_start, output_video, mode):
        if mode == 'CAT':
            output = torch.cat([filter_start.unsqueeze(dim=1).repeat(1,output_video.shape[1],1),output_video],dim=-1)
        elif mode == 'COS':
            output = filter_start.unsqueeze(dim=1).repeat(1,output_video.shape[1],1) * output_video  
        elif mode =='SUB':
            output = (filter_start.unsqueeze(dim=1).repeat(1,output_video.shape[1],1) - output_video)
        elif mode == 'CROSS_COS':
            output = filter_start * output_video
        elif mode == 'CROSS_SUB':
            output = torch.abs(filter_start - output_video)
        return output

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

    def mask_softmax(self, feat, mask):
        return self.masked_softmax(feat, mask, memory_efficient=False)

    def kl_div(self, p, gt, length):
        individual_loss = []
        for i in range(length.size(0)):
            vlength = int(length[i])
            ret = gt[i][:vlength] * torch.log(p[i][:vlength]/gt[i][:vlength])
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss
    



    def max_boundary(self, p, gt, length):
        individual_loss = []
        for i in range(length.size(0)):
            # vlength = int(length[i])
            index_bd = gt[i]
            ret = torch.log(p[i][index_bd])
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def max_inter(self, p, gt_s, gt_e, length):
        individual_loss = []
        for i in range(length.size(0)):
            # vlength = int(length[i])
            index_bs = gt_s[i]
            index_be = gt_e[i]
            ret = torch.log(p[i][index_bs:(index_be+1)])/(max(index_be-index_bs,1))
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def forward(self, videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz, frame_start, frame_end):

        mask = self.get_mask_from_sequence_lengths(videoFeat_lengths, int(videoFeat.shape[1]))

        output_video = self.model_video_GRU(videoFeat,videoFeat_lengths,mask)
        filter_start, lengths = self.model_df(tokens, tokens_lengths,output_video)
        # output_video =  self.feature_gauss_normalize(output_video)
        # filter_start = self.feature_gauss_normalize(filter_start)
        # attention_weights = attention_weights.detach().cpu().numpy()
        # np.save('/home/thy/disk/proposal_free/experiments/visualization/attention.npy',attention_weights)

        output = self.fusion_layer(filter_start,output_video,self.cfg.ACRM_CLASSIFICATION.FUSION)
        # output = torch.cat([filter_start.unsqueeze(dim=1).repeat(1,output_video.shape[1],1),output_video],dim=-1)
        # output = filter_start.unsqueeze(dim=1).repeat(1,output_video.shape[1],1) * output_video
        if self.is_use_rnn_loc == True:
            output, _ = feed_forward_rnn(self.rnn_localization,
                            output,
                            lengths=videoFeat_lengths)
            output = self.dropout_layer(output)
        pred_start = self.starting(output.view(-1, output.size(2))).view(-1,output.size(1),1).squeeze()
        pred_start = self.mask_softmax(pred_start, mask)

        pred_end = self.ending(output.view(-1, output.size(2))).view(-1,output.size(1),1).squeeze()
        pred_end = self.mask_softmax(pred_end, mask)

        pred_inter = self.intering(output.view(-1, output.size(2))).view(-1,output.size(1),1).squeeze()
        pred_inter = self.mask_softmax(pred_inter, mask)

        start_loss, individual_start_loss = self.max_boundary(pred_start, frame_start, videoFeat_lengths)
        end_loss, individual_end_loss     = self.max_boundary(pred_end, frame_end, videoFeat_lengths)
        inter_loss, individual_inter_loss = self.max_inter(pred_inter,frame_start,frame_end,videoFeat_lengths)

        individual_loss = individual_start_loss + individual_end_loss + individual_inter_loss
        atten_loss = torch.tensor(0).cuda()
        # atten_loss = torch.sum(-( (1-localiz) * torch.log((1-attention) + 1E-12)), dim=1)
        # atten_loss = torch.mean(atten_loss)
        attention = output_video[:,:,0]
        if True:
            # total_loss = start_loss + end_loss + atten_loss
            total_loss = start_loss + end_loss + 1*inter_loss
        else:
            total_loss = start_loss + end_loss

        return total_loss, individual_loss, pred_start, pred_end, attention, atten_loss

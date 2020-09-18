import torch
import numpy as np
from utils import rnns


class BatchCollator(object):

    def __call__(self, batch):
        # print(batch)
        transposed_batch = list(zip(*batch))
        index     = transposed_batch[0]
        videoFeat = transposed_batch[1]
        tokens    = transposed_batch[2]
        start = transposed_batch[3]
        end = transposed_batch[4]
        localiz = transposed_batch[5]
        time_start = transposed_batch[6]
        time_end = transposed_batch[7]
        factor = transposed_batch[8]
        fps = transposed_batch[9]
        frame_start= torch.tensor([i for i in transposed_batch[10]])
        frame_end = torch.tensor([i for i in transposed_batch[11]])
        duration = transposed_batch[12]
        vid_names = transposed_batch[13]
        # duration = torch.tensor([i for i in transposed_batch[12]]).float()

        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat)
        localiz, localiz_lengths = rnns.pad_sequence(localiz)
        tokens, tokens_lengths   = rnns.pad_sequence(tokens)

        start, start_lengths = rnns.pad_sequence(start)
        end, end_lengths     = rnns.pad_sequence(end)
        # torch.tensor([])

        return index, \
               videoFeat, \
               videoFeat_lengths, \
               tokens, \
               tokens_lengths, \
               start,  \
               end, \
               localiz, \
               localiz_lengths, \
               time_start, \
               time_end, \
               factor, \
               fps,\
               frame_start,\
               frame_end,\
               duration,\
               vid_names

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.pe[:, :x.size(1)]


class VIDEOEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of VIDEOEmbedding
    """

    def __init__(self, video_dim, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super(VIDEOEmbedding,self).__init__()
        #self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=embed_size)
        self.linear = nn.Linear(video_dim,embed_size) 
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x1 = self.linear(sequence)  
        # print(x1.shape)
        x2 = self.position(sequence)
        return self.dropout(x1+x2)



class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))




class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, h, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
    """
    #(batch_size, -1, self.h, self.d_k).transpose(1, 2) ==>(batch_size,  self.h, -1, self.d_k)
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h == 0
        ## d_model = h * d_k
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)


        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        # return torch.matmul(p_attn, value), p_attn
        scores = torch.matmul(scores, value)



        # 3) "Concat" using a view and apply a final linear.
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        scores = self.output_linear(scores)

        scores = self.norm(scores + self.dropout(scores))

        return scores

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        #cross_hidden, cross_attn_heads, cross_hidden * 4, dropout (12,12,768*4,0.1)
        """

        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.attention(x, x, x, mask=mask)
        x = self.output_sublayer(x, self.feed_forward)
        # return self.dropout(x)
        return x


class Transformer(nn.Module):
    """
    CBT_BASE model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, cfg):
        #CBT_BASE(len(vocab), video_hidden=args.video_hidden, video_n_layers=args.layers, video_attn_heads=args.attn_heads)

        """
        :param vocab_size: vocab_size of total words
        :param video_hidden: CBT_BASE model video_hidden size(video_transformer)
        :param video_n_layers: numbers of Transformer blocks(layers)
        :param video_attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(Transformer,self).__init__()
        self.video_hidden = cfg.DYNAMIC_FILTER.VIDEO_MULTIHEAD.EMBED_DIM
        self.video_n_layers = cfg.DYNAMIC_FILTER.VIDEO_MULTIHEAD.NUM_LAYERS
        self.video_attn_heads = cfg.DYNAMIC_FILTER.VIDEO_MULTIHEAD.NUM_HEADS

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.video_hidden * 4
        self.dropout = cfg.DYNAMIC_FILTER.VIDEO_MULTIHEAD.DROPOUT
        # self.reduction  = nn.Linear()
        if cfg.DATASETS.TRAIN == 'anet_cap_train':
            cfg.REDUCTION.INPUT_SIZE = 500
        self.video_embedding = VIDEOEmbedding(cfg.REDUCTION.INPUT_SIZE, cfg.REDUCTION.OUTPUT_SIZE)
        assert cfg.DYNAMIC_FILTER.VIDEO_MULTIHEAD.EMBED_DIM == cfg.REDUCTION.OUTPUT_SIZE


        # multi-layers transformer blocks, deep network
        self.video_transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.video_hidden, self.video_attn_heads, self.feed_forward_hidden, self.dropout) for _ in range(self.video_n_layers)])

    def forward(self, video_input, lengths = None , video_mask= None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (text_input > 0).unsqueeze(1).repeat(1, text_input.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        #text_input = self.embedding(text_input, segment_info)
        #text_output = self.text_BERT_base(text_input)
    
        # running over multiple transformer blocks
        video_mask = video_mask.view(video_input.shape[0],1,-1,1).repeat(1,self.video_attn_heads,1,1).contiguous().float()
        video_mask = torch.matmul(video_mask,video_mask.permute(0,1,3,2).contiguous()).long()
        video_output = self.video_embedding(video_input)
        for transformer in self.video_transformer_blocks:
            video_output = transformer.forward(video_output,video_mask)
        
        #cross_input = torch.cat([text_output,video_output],dim=1)

        #for transformer in self.cross_transformer_blocks:
            #cross_output = transformer.forward(cross_input, mask)

        return video_output, lengths
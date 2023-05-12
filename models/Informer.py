import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.delay import TemporalDecay
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer, DelayAttention
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Imputation
        self.enc_decay = TemporalDecay(d_in=configs.enc_in, d_out=configs.d_model, diag=False)
        self.dec_decay = TemporalDecay(d_in=configs.dec_in, d_out=configs.d_model, diag=False)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        Attn = ProbAttention
        if attn == 'prob':
            Attn = ProbAttention
        elif attn == 'delay':
            Attn = DelayAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_mask, dec_mask,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoder gamma
        delta_fwd = TemporalDecay.compute_delta_forward(enc_mask)
        delta_bwd = TemporalDecay.compute_delta_backward(enc_mask)
        enc_delta = TemporalDecay.compute_delta(delta_fwd, delta_bwd)
        enc_gamma = self.enc_decay(enc_delta.to(x_enc.device)).to(x_enc.device)  # [B, H, L, D]

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, gamma=enc_gamma)
        # decoder gamma
        delta_fwd = TemporalDecay.compute_delta_forward(dec_mask)
        delta_bwd = TemporalDecay.compute_delta_backward(dec_mask)
        dec_delta = TemporalDecay.compute_delta(delta_fwd, delta_bwd)
        dec_gamma = self.dec_decay(dec_delta.to(x_dec.device)).to(x_dec.device)  # [B, H, L, D]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, gamma=dec_gamma)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

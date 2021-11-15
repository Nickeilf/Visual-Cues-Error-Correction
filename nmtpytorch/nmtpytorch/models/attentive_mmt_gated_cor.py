# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..metrics import Metric
from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, TextEncoder, FF, TextGating
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class GatedAttentiveMMTCOR(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'fusion_activ': 'tanh',     # Multimodal context non-linearity
            'vis_activ': 'linear',      # Visual feature transformation activ.
            'n_channels': 2048,         # depends on the features used
            'mm_att_type': 'md-dd',     # multimodal attention type
                                        # md: modality dep.
                                        # mi: modality indep.
                                        # dd: decoder state dep.
                                        # di: decoder state indep.
            'out_logic': 'deep',        # simple vs deep output
            'persistent_dump': False,   # To save activations during beam-search
            'preatt': False,            # Apply filtered attention
            'preatt_activ': 'ReLU',     # Activation for convatt block
            'dropout_img': 0.0,         # Dropout on image features
            'gating_type': 'mlp',
            'gating_dim': 512,
            'dropout_gate': 0,
        })

    def __init__(self, opts):
        super().__init__(opts)

        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)
        if tlangs:
            print(tlangs[0])
            print(tlangs[1])
            self.tls = [tlang for tlang in tlangs]
            self.trg_vocabs = {tl: self.vocabs[tl] for tl in self.tls}
            self.n_trg_vocabs = {tl: len(self.trg_vocabs[tl]) for tl in self.tls}
            # Need to be set for early-stop evaluation
            # NOTE: This should come from config or elsewhere
            self.val_refss = {tl: self.opts.data['val_set'][tl] for tl in self.tls}

        self.sigma = opts.train['sigma']

    def setup(self, is_train=True):
        super().setup(is_train)

        # Textual context dim
        txt_ctx_size = self.ctx_sizes[self.sl]

        # Add visual context transformation (sect. 3.2 in paper)
        self.ff_img = FF(
            self.opts.model['n_channels'], txt_ctx_size,
            activ=self.opts.model['vis_activ'])

        self.dropout_img = nn.Dropout(self.opts.model['dropout_img'])

        # Add vis ctx size
        self.ctx_sizes['image'] = txt_ctx_size

        ########################
        # Create Textual Encoder
        ########################
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Create Decoder
        self.dec = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocabs[self.tls[0]],
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            fusion_activ=self.opts.model['fusion_activ'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            mm_att_type=self.opts.model['mm_att_type'],
            out_logic=self.opts.model['out_logic'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            att_ctx2hid=False,
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            persistent_dump=self.opts.model['persistent_dump'])

        self.dec_cor = ConditionalMMDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocabs[self.tls[1]],
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            fusion_activ=self.opts.model['fusion_activ'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            mm_att_type=self.opts.model['mm_att_type'],
            out_logic=self.opts.model['out_logic'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            att_ctx2hid=False,
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            persistent_dump=self.opts.model['persistent_dump'])

        self.gating = TextGating(text_dim=txt_ctx_size,
                                 feat_dim=txt_ctx_size,
                                 mid_dim=self.opts.model['gating_dim'],
                                 dropout=self.opts.model['dropout_gate'],
                                 att_activ=self.opts.model['att_activ'],
                                 gating_type=self.opts.model['gating_type'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            order_file=self.opts.data[split + '_set'].get('ord', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Transform the features to context dim
        feats = self.dropout_img(self.ff_img(batch['feats']))

        # Get source language encodings (S*B*C)
        text_encoding = self.enc(batch[self.sl])

        # SxBxC -> CxSxB
        text_mask = text_encoding[1]
        text_hidden = text_encoding[0].permute(2, 0, 1)

        text_gate = self.gating(feats, text_encoding[0])
        text_hidden = text_gate * text_hidden
        text_hidden = text_hidden.permute(1, 2, 0)
        text_encoding = (text_hidden, text_mask)

        return {
            str(self.sl): text_encoding,
            'image': (feats, None),
        }

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        # Get loss dict
        result_mt = self.dec(self.encode(batch), batch[self.tls[0]])
        result_cor = self.dec_cor(self.encode(batch), batch[self.tls[1]])

        # weighted loss
        result_cor['loss'] *= self.sigma

        result_mt['n_items'] = torch.nonzero(batch[self.tls[0]][1:]).shape[0]
        result_cor['n_items'] = torch.nonzero(batch[self.tls[1]][1:]).shape[0]
        return {'MT': result_mt,
                'COR': result_cor}

    def get_decoder(self, task_id=None):
        """Compatibility function for multi-tasking architectures."""
        if 'cor' in task_id:
            return self.dec_cor
        else:
            return self.dec

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch)
            loss.store_multi_loss(out)

            for idx, tid in enumerate(out):
                if idx == 0:
                    loss.update(out[tid]['loss'], out[tid]['n_items'])
                else:
                    loss.update(out[tid]['loss'], out[tid]['n_items'], task2=True)

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

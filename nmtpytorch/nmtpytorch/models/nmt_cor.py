# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..metrics import Metric
from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, TextEncoder, FF
from ..layers.decoders import get_decoder
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class NMTCOR(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()

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

        Decoder = get_decoder(self.opts.model['dec_variant'])
        self.dec_cor = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocabs[self.tls[1]],
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            bos_type=self.opts.model['bos_type'],
            bos_dim=self.opts.model['bos_dim'],
            bos_activ=self.opts.model['bos_activ'],
            bos_bias=self.opts.model['bos_type'] == 'feats',
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'])

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

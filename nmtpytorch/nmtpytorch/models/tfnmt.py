# -*- coding: utf-8 -*-
import logging

import torch
import math
from torch import nn

from ..datasets import MultimodalDataset
from ..layers import TFEncoder, TFDecoder
from ..utils.nn import LabelSmoothingLoss

from ..utils.topology import Topology
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions

from . import NMT

logger = logging.getLogger('nmtpytorch')


class TransformerNMT(NMT):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'model_dim': 512,           # Source and target embedding sizes,
            'num_heads': 8,             # The number of attention heads
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'short_list': 0,            # Short list vocabularies (0: disabled)
            'enc_n_layers': 6,          # The number of encoder layers
            'dec_n_layers': 6,          # The number of decoder layers
            'enc_ff_dim': 2048,         # The number of encoder feed forward dimensions
            'dec_ff_dim': 2048,         # The number of decoder feed forward dimensions
            'enc_bidirectional': True,  # Whether the encoder is bidirectional or unidirectional.
            'tied_emb': False,          # Whether the embedding should be tied.
            'ff_activ': 'gelu',         # The feed forward layer activation function. Default 'gelu'.
            'dropout': 0.1,             # The dropout.
            'attn_dropout': 0.0,        # The attention dropout.
            'pre_norm': True,           # Indicates whether to use pre_norm (recent) or post_norm (original) layers.
            # Visual features (optional)
            'feat_mode': None,
            'aux_dim': None,            # Auxiliary features dim (# channels for conv features)
            'aux_dropout': 0.0,         # Auxiliary features dropout
            'aux_lnorm': False,         # layer-norm
            'aux_l2norm': False,        # L2-normalize
            'aux_proj_dim': None,       # Projection layer for features
            'aux_proj_activ': None,     # Projection layer non-linearity
            'img_boxes_dim': None,      # The vector dimension for the boxes, associated with a region.
            'num_regions': 36,          # The number of regions to use. Valid only for OD features. Default: 36.
            'mm_fusion_op': None,       # fusion type
            'mm_fusion_dropout': 0.0,   # fusion dropout
            'tf_dec_img_attn': None,    # The decoder visual attention; could be: 'serial', 'parallel' or None.
            'tf_n_mm_hier_heads': 8,    # Used with hierarchical image attention to specify the number of hierarchical heads. Default 8.
                                        # Default: None.
            # Decoding/training simultaneous NMT args
            'translator_type': 'gs',   # This model implements plain unidirectional MT
                                        # so the decoding is normal greedy-search
            'translator_args': {},      # No extra arguments to translator
        }

    def __init__(self, opts):
        super().__init__(opts)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        with torch.no_grad():
            self.enc.src_embedding.weight.data[0].fill_(0)
            self.dec.trg_embedding.weight.data[0].fill_(0)

    def setup(self, is_train=True):
        # super().setup(is_train)

        self.enc = TFEncoder(
            model_dim=self.opts.model["model_dim"],
            n_heads=self.opts.model["num_heads"],
            ff_dim=self.opts.model["enc_ff_dim"],
            n_layers=self.opts.model["enc_n_layers"],
            num_embeddings=self.n_src_vocab,
            ff_activ=self.opts.model["ff_activ"],
            dropout=self.opts.model["dropout"],
            attn_dropout=self.opts.model["attn_dropout"],
            pre_norm=self.opts.model["pre_norm"],
            enc_bidirectional=self.opts.model["enc_bidirectional"]
        )

        self.dec = TFDecoder(
            model_dim=self.opts.model["model_dim"],
            n_heads=self.opts.model["num_heads"],
            ff_dim=self.opts.model["dec_ff_dim"],
            n_layers=self.opts.model["dec_n_layers"],
            num_embeddings=self.n_trg_vocab,
            tied_emb=self.opts.model["tied_emb"],
            ff_activ=self.opts.model["ff_activ"],
            dropout=self.opts.model["dropout"],
            attn_dropout=self.opts.model["attn_dropout"],
            pre_norm=self.opts.model["pre_norm"],
            img_attn=self.opts.model["tf_dec_img_attn"],
            n_mm_hier_heads=self.opts.model["tf_n_mm_hier_heads"],
        )

        self.loss = LabelSmoothingLoss(
            trg_vocab_size=self.n_trg_vocab, reduction='sum', ignore_index=0,
            with_logits=False)

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            assert self.n_src_vocab == self.n_trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."
            self.enc.src_embedding.weight = self.dec.trg_embedding.weight

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
        h, mask = self.enc(batch[self.sl])

        d = {str(self.sl): (h, mask)}
        return d

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """ 
        encoded_src = self.encode(batch)

        # The input to the transformer should include the <bos> token but not the <eos> token.
        target_input = batch[self.tl][:-1, :]

        # The actual values should not have the <bos> token but should include the <eos>
        target_real = batch[self.tl][1:, :]

        result, _ = self.dec(encoded_src, target_input, **kwargs)

        total_loss = self.loss(
            result.contiguous().view(-1, result.size(-1)), target_real.contiguous().view(-1))

        return {
            'loss': total_loss,
            'n_items': target_real.nonzero(as_tuple=False).size(0),
        }
    
    @staticmethod
    def beam_search(models, data_loader, task_id=None, beam_size=12, max_len=200,
                    lp_alpha=0., suppress_unk=False, n_best=False):
        """An efficient implementation for beam-search algorithm.

        Arguments:
            models (list of Model): Model instance(s) derived from `nn.Module`
                defining a set of methods. See `models/nmt.py`.
            data_loader (DataLoader): A ``DataLoader`` instance.
            task_id (str, optional): For multi-output models, this selects
                the decoder. (Default: None)
            beam_size (int, optional): The size of the beam. (Default: 12)
            max_len (int, optional): Maximum target length to stop beam-search
                if <eos> is still not generated. (Default: 200)
            lp_alpha (float, optional): If > 0, applies Google's length-penalty
                normalization instead of simple length normalization.
                lp: ((5 + |Y|)^lp_alpha / (5 + 1)^lp_alpha)
            suppress_unk (bool, optional): If `True`, suppresses the log-prob
                of <unk> token.
            n_best (bool, optional): If `True`, returns n-best list of the beam
                with the associated scores.

        Returns:
            list:
                A list of hypotheses in surface form.
        """
        def tile_ctx_dict(ctx_dict, idxs):
            """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
            # 1st: tensor, 2nd optional mask
            return {
                k: (t[:, idxs], None if mask is None else mask[idxs, :])
                for k, (t, mask) in ctx_dict.items()
            }

        def check_context_ndims(ctx_dict):
            for name, (ctx, mask) in ctx_dict.items():
                assert ctx.dim() == 3, \
                    f"{name}'s 1st dim should always be a time dimension."

        def decoder_step(model, state_dict, next_word_idxs, h, hypothesis=None):
            logp, h = model.dec.f_next(
                state_dict, model.dec.get_emb(next_word_idxs), h, hypothesis)

            # Similar to the logic in fairseq https://bit.ly/3agXAa7
            # Never select the pad token or the bos token
            logp[:, model.trg_vocab['<pad>']] = -math.inf
            logp[:, model.trg_vocab['<bos>']] = -math.inf

            # Compute most likely word idxs
            next_word_idxs = logp.argmax(dim=-1)
            return logp, h, next_word_idxs

        # This is the batch-size requested by the user but with sorted
        # batches, efficient batch-size will be <= max_batch_size
        max_batch_size = data_loader.batch_sampler.batch_size
        k = beam_size
        inf = -1000
        results = []
        enc_args = {}

        
        if task_id is None:
            # For classical models that have single encoder, decoder and
            # target vocabulary
            decs = [m.dec for m in models]
            f_inits = [dec.f_init for dec in decs]
            f_nexts = [dec.f_next for dec in decs]
            vocab = models[0].trg_vocab
        else:
            # A specific input-output topology has been requested
            task = Topology(task_id)
            enc_args['enc_ids'] = task.srcs
            # For new multi-target models: select the first target decoder
            decs = [m.get_decoder(task.first_trg) for m in models]
            # Get the necessary init() and next() methods
            f_inits = [dec.f_init for dec in decs]
            f_nexts = [dec.f_next for dec in decs]
            # Get the corresponding vocabulary for the first target
            vocab = models[0].vocabs[task.first_trg]

        # Common parts
        encoders = [m.encode for m in models]
        unk = vocab['<unk>']
        eos = vocab['<eos>']
        n_vocab = len(vocab)


        # Only for single model
        model = models[0]
        enc = encoders[0]
        dec = decs[0]
        f_init = f_inits[0]
        f_next = f_nexts[0]

        # Tensorized beam that will shrink and grow up to max_batch_size
        beam_storage = torch.zeros(
            max_len, max_batch_size, k, dtype=torch.long, device=DEVICE)
        mask = torch.arange(max_batch_size * k, device=DEVICE)
        nll_storage = torch.zeros(max_batch_size, device=DEVICE)

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)

            # Always use the initial storage
            beam = beam_storage.narrow(1, 0, batch.size).zero_()

            # Mask to apply to pdxs.view(-1) to fix indices
            nk_mask = mask.narrow(0, 0, batch.size * k)

            # nll: batch_size x 1 (will get expanded further)
            nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1)

            # Tile indices to use in the loop to expand first dim
            tile = range(batch.size)

            # Encode source modalities
            # ctx_dicts = [encode(batch, **enc_args) for encode in encoders]
            state_dict = enc(batch, **enc_args)

            # Sanity check one of the context dictionaries for dimensions
            # check_context_ndims(ctx_dicts[0])
            check_context_ndims(state_dict)

            # Get initial decoder state (N*H)
            # h_ts = [f_init(ctx_dict) for f_init, ctx_dict in zip(f_inits, ctx_dicts)]
            h_t = f_init(state_dict)

            # we always have <bos> tokens except that the returned embeddings
            # may differ from one model to another.
            # idxs = models[0].get_bos(batch.size).to(DEVICE)
            next_word_idxs = model.get_bos(batch.size).to(DEVICE)

            # The Transformer decoder require the <bos> to be passed alongside all hypothesis objects for prediction
            tf_decoder_input = next_word_idxs.unsqueeze(0)

            first_step = True
            for tstep in range(max_len):
                # Select correct positions from source context
                # ctx_dicts = [tile_ctx_dict(cd, tile) for cd in ctx_dicts]

                state_dict = tile_ctx_dict(state_dict, tile)

                # Get log probabilities and next state
                # log_p: batch_size x vocab_size (t = 0)
                #        batch_size*beam_size x vocab_size (t > 0)
                # NOTE: get_emb does not exist in some models, fix this.
                # log_ps, h_ts = zip(
                #     *[f_next(cd, dec.get_emb(idxs, tstep), h_t[tile]) for
                #       f_next, dec, cd, h_t in zip(f_nexts, decs, ctx_dicts, h_ts)])

                # Do the actual averaging of log-probabilities
                # log_p = sum(log_ps).data

                log_p, h_t, next_word_idxs = decoder_step(model, state_dict, next_word_idxs, h_t, tf_decoder_input)

                if suppress_unk:
                    log_p[:, unk] = inf

                # Detect <eos>'d hyps
                next_word_idxs = (next_word_idxs == 2).nonzero()
                if next_word_idxs.numel():
                    if next_word_idxs.numel() == batch.size * k:
                        break
                    next_word_idxs.squeeze_(-1)
                    # Unfavor all candidates
                    log_p.index_fill_(0, next_word_idxs, inf)
                    # Favor <eos> so that it gets selected
                    log_p.view(-1).index_fill_(0, next_word_idxs * n_vocab + 2, 0)

                # Expand to 3D, cross-sum scores and reduce back to 2D
                # log_p: batch_size x vocab_size ( t = 0 )
                #   nll: batch_size x beam_size (x 1)
                # nll becomes: batch_size x beam_size*vocab_size here
                # Reduce (N, K*V) to k-best
                nll, beam[tstep] = nll.unsqueeze_(2).add(log_p.view(
                    batch.size, -1, n_vocab)).view(batch.size, -1).topk(
                        k, sorted=False, largest=True)

                # previous indices into the beam and current token indices
                pdxs = beam[tstep] / n_vocab
                beam[tstep].remainder_(n_vocab)
                next_word_idxs = beam[tstep].view(-1)

                
                # Compute correct previous indices
                # Mask is needed since we're in flattened regime
                tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)
                
                if first_step:
                    first_step = False
                    tf_decoder_input = model.get_bos(batch.size*k).to(DEVICE).unsqueeze(0)

                # Add the predicted word to the decoder's input. Used for the transformer models.
                tf_decoder_input = torch.cat((tf_decoder_input, next_word_idxs.unsqueeze(0)), dim=0)

                if tstep > 0:
                    # Permute all hypothesis history according to new order
                    beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))

            # Put an explicit <eos> to make idxs_to_sent happy
            beam[max_len - 1] = eos

            # Find lengths by summing tokens not in (pad,bos,eos)
            len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

            if lp_alpha > 0.:
                len_penalty = ((5 + len_penalty)**lp_alpha) / 6**lp_alpha

            # Apply length normalization
            nll.div_(len_penalty)

            if n_best:
                # each elem is sample, then candidate
                tbeam = beam.permute(1, 2, 0).to('cpu').tolist()
                scores = nll.to('cpu').tolist()
                results.extend(
                    [(vocab.list_of_idxs_to_sents(b), s) for b, s in zip(tbeam, scores)])
            else:
                # Get best-1 hypotheses
                top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
                hyps = beam[:, range(batch.size), top_hyps].t().to('cpu')
                results.extend(vocab.list_of_idxs_to_sents(hyps.tolist()))

        # Recover order of the samples if necessary
        return sort_predictions(data_loader, results)

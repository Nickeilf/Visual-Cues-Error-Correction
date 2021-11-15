import logging
import math

import torch

from ..utils.device import DEVICE
from ..utils.io import progress_bar
from ..utils.data import sort_predictions
from ..models import SimultaneousTFNMT

logger = logging.getLogger('pysimt')


"""Batched vanilla beam search without any simultaneous translation
features."""

class BeamSearch:
    def __init__(self, model, data_loader, out_prefix, batch_size, filter_chain=None,
                 max_len=100, beam_size=12, lp_alpha=0., suppress_unk=False, n_best=False, **kwargs):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.filter_chain = filter_chain
        self.out_prefix = out_prefix
        self.beam_size = beam_size
        self.lp_alpha = lp_alpha
        self.suppress_unk = suppress_unk
        self.n_best = n_best

        if isinstance(self.model, SimultaneousTFNMT):
            self.tf=True
        else:
            self.tf=False

        self.vocab = self.model.trg_vocab
        self.n_vocab = len(self.vocab)
        self.unk = self.vocab['<unk>']
        self.eos = self.vocab['<eos>']
        self.bos = self.vocab['<bos>']
        self.pad = self.vocab['<pad>']

        self.max_len = max_len
        self.do_dump = out_prefix != ''

    def dump_results(self, hyps, suffix=''):
        suffix = 'beam' if not suffix else f'{suffix}.beam'
        suffix += str(self.beam_size)

        # Dump raw ones (BPE/SPM etc.)
        self.dump_lines(hyps, suffix + '.raw')
        if self.filter_chain is not None:
            self.dump_lines(self.filter_chain.apply(hyps), suffix)

    def dump_lines(self, lines, suffix):
        fname = f'{self.out_prefix}.{suffix}'
        with open(fname, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')

    def decoder_step(self, state_dict, next_word_idxs, h, hypothesis=None):
        logp, h = self.model.dec.f_next(
            state_dict, self.model.dec.get_emb(next_word_idxs), h, hypothesis)

        # Similar to the logic in fairseq https://bit.ly/3agXAa7
        # Never select the pad token or the bos token
        logp[:, self.pad] = -math.inf
        logp[:, self.bos] = -math.inf

        # Compute most likely word idxs
        next_word_idxs = logp.argmax(dim=-1)
        return logp, h, next_word_idxs

    def decoder_init(self, state_dict=None):
        return self.model.dec.f_init(state_dict)

    def run_all(self):
        return self.run()

    def run(self, **kwargs):
        # effective batch size may be different
        max_batch_size = self.data_loader.batch_sampler.batch_size
        k = self.beam_size
        inf = -1000
        results = []

        # Common parts
        unk = self.unk
        eos = self.eos
        n_vocab = self.n_vocab
        


        for batch in progress_bar(self.data_loader, unit='batch'):
            batch.device(DEVICE)
            bsz = batch.size

            # initialize buffers
            scores = (
                torch.zeros(bsz * k, self.max_len + 1).to(DEVICE).float()
            )  # +1 for eos; pad is never chosen for scoring
            tokens = (
                torch.zeros(bsz * k, self.max_len + 2)
                .to(DEVICE)
                .long()
                .fill_(self.pad)
            )  # +2 for eos and pad
            tokens[:, 0] = self.eos if self.bos is None else self.bos

            cands_to_ignore = (
                torch.zeros(bsz, k).to(DEVICE).eq(-1)
            )

            

            # a boolean array indicating if the sentence at the index is finished or not
            finished = [False for i in range(bsz)]
            num_remaining_sent = bsz  # number of sentences remaining

            # number of candidate hypos per step
            cand_size = 2 * k  # 2 x beam size in case half are EOS

            # offset arrays for converting between different indexing schemes
            bbsz_offsets = (
                (torch.arange(0, bsz) * k)
                .unsqueeze(1)
                .type_as(tokens)
                .to(DEVICE)
            )

            cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(DEVICE)

            # Tile indices to use in the loop to expand first dim
            tile = range(batch.size)

            # Cache encoder states
            self.model.cache_enc_states(batch)

            # Get encoder hidden states
            state_dict = self.model.get_enc_state_dict()

            # Sanity check one of the context dictionaries for dimensions
            check_context_ndims(state_dict)

            # Initial state is None i.e. 0. state_dict is not used
            h = self.decoder_init(state_dict)

            # last batch could be smaller than the requested batch size
            cur_batch_size = batch.size

            # Start all sentences with <s>
            next_word_idxs = self.model.get_bos(cur_batch_size).to(DEVICE)

            # The Transformer decoder require the <bos> to be passed alongside all hypothesis objects for prediction
            tf_decoder_input = next_word_idxs.unsqueeze(0)

            # tf_decoder_input shape [batch_size] at first step, and [batch_size * k] after
            first_step = True
            for t in range(self.max_len):
                state_dict = tile_ctx_dict(state_dict, tile, tf=self.tf)

                logp, h, next_word_idxs = self.decoder_step(
                    state_dict, next_word_idxs, h, tf_decoder_input)

                if self.suppress_unk:
                    logp[:, unk] = inf

                # Detect <eos>'d hyps
                next_word_idxs = (next_word_idxs == 2).nonzero()
                if next_word_idxs.numel():
                    if next_word_idxs.numel() == batch.size * k:
                        break
                    next_word_idxs.squeeze_(-1)
                    # Unfavor all candidates
                    logp.index_fill_(0, next_word_idxs, inf)
                    # Favor <eos> so that it gets selected
                    logp.view(-1).index_fill_(0, next_word_idxs * n_vocab + 2, 0)
                
                # Expand to 3D, cross-sum scores and reduce back to 2D
                # log_p: batch_size x vocab_size ( t = 0 )
                #   nll: batch_size x beam_size (x 1)
                # nll becomes: batch_size x beam_size*vocab_size here
                # Reduce (N, K*V) to k-best
                nll, beam[t] = nll.unsqueeze_(2).add(logp.view(
                    batch.size, -1, n_vocab)).view(batch.size, -1).topk(
                        k, sorted=False, largest=True)
                    
                # previous indices into the beam and current token indices
                pdxs = beam[t] / n_vocab
                beam[t].remainder_(n_vocab)
                next_word_idxs = beam[t].view(-1)

                if first_step:
                    first_step = False
                    tf_decoder_input = self.model.get_bos(batch.size*k).to(DEVICE).unsqueeze(0)

                # Add the predicted word to the decoder's input. Used for the transformer models.
                tf_decoder_input = torch.cat((tf_decoder_input, next_word_idxs.unsqueeze(0)), dim=0)

                # Compute correct previous indices
                # Mask is needed since we're in flattened regime
                tile = pdxs.view(-1) + (nk_mask / k) * (k if t else 1)

                if t > 0:
                    # Permute all hypothesis history according to new order
                    beam[:t] = beam[:t].gather(2, pdxs.repeat(t, 1, 1))

            # Put an explicit <eos> to make idxs_to_sent happy
            beam[self.max_len - 1] = eos

            # Find lengths by summing tokens not in (pad,bos,eos)
            len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

            if self.lp_alpha > 0.:
                len_penalty = ((5 + len_penalty)**self.lp_alpha) / 6**self.lp_alpha

            # Apply length normalization
            nll.div_(len_penalty)

            if self.n_best:
                # each elem is sample, then candidate
                tbeam = beam.permute(1, 2, 0).to('cpu').tolist()
                scores = nll.to('cpu').tolist()
                results.extend(
                    [(self.vocab.list_of_idxs_to_sents(b), s) for b, s in zip(tbeam, scores)])
            else:
                # Get best-1 hypotheses
                top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
                hyps = beam[:, range(batch.size), top_hyps].t().to('cpu')
                results.extend(self.vocab.list_of_idxs_to_sents(hyps.tolist()))

        hyps = sort_predictions(self.data_loader, results)
        if self.do_dump:
            self.dump_results(hyps)

        return (hyps,)
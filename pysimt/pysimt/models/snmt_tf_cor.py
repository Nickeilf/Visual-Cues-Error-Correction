import torch
import math
from torch import nn

from ..layers import TFEncoder, TFDecoder
from ..utils.nn import LabelSmoothingCELoss, LabelSmoothingLoss
from ..utils.ml_metrics import Loss
from ..utils.io import progress_bar
from ..utils.device import DEVICE
from ..metrics import Metric
from . import SimultaneousNMT


class SimultaneousTFNMTCOR(SimultaneousNMT):

    def __init__(self, opts):
        """
        Creates a SimultaneousNMTTransformer.
        :param opts: The options.
        """
        self.defaults = None
        super().__init__(opts)

        # These will be initialized in the setup method, similarly to other models.
        self.encoders = {}
        self.dec = None
        self.loss = None
        self.current_batch = None
        self.sigma = opts.train['sigma']

        self.cor_vocab = self.vocabs["src_cor"]
        self.n_cor_vocab = len(self.cor_vocab)

    def setup(self, is_train=True):
        """
        Initialises the necessary model components.
        :param is_train: Whether the model is in training mode or not.
        """
        encoders = {}
        for key in self.topology.srcs.keys():
            encoders[key] = getattr(self, f'_create_{key}_encoder')()
        self.encoders = nn.ModuleDict(encoders)
        self.dec = self._create_decoder()
        self.dec_cor = self._create_cor_decoder()

        assert id(self.dec) != id(self.dec_cor)

        self.loss = LabelSmoothingLoss(
            trg_vocab_size=self.n_trg_vocab, label_smoothing=0.1, reduction='sum', ignore_index=0,
            with_logits=False)

        self.cor_loss = LabelSmoothingLoss(
            trg_vocab_size=self.n_cor_vocab, label_smoothing=0.1, reduction='sum', ignore_index=0,
            with_logits=False)

        # self.loss = LabelSmoothingCELoss()
        # self.loss = nn.NLLLoss(reduction="sum", ignore_index=0)

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            assert self.n_src_vocab == self.n_trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."
            self.encoders[str(self.sl)].src_embedding.weight = self.dec.trg_embedding.weight

    def reset_parameters(self):
        """
        Initialize the model parameters.
        """
        # changed
        for param in self.parameters():
            if param.requires_grad and param.dim() > 1:
                # nn.init.xavier_normal_(param.data, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(param)

        # Reset padding embedding to 0
        for key, enc in self.encoders.items():
            if hasattr(enc, 'src_embedding'):
                # d = 0.01 # FixNorm
                # nn.init.uniform_(enc.src_embedding.weight, a=-d, b=d) #FixNorm

                # fairseq
                nn.init.normal_(enc.src_embedding.weight, mean=0, std=self.opts.model["model_dim"] ** -0.5)
                nn.init.constant_(enc.src_embedding.weight[0], 0)

                # enc.src_embedding.load_pretrained_vectors("/export/vol/gluster_lama_quetzal/zli/MMT-noise/glove300.pt")


        nn.init.normal_(self.dec.trg_embedding.weight, mean=0, std=self.opts.model["model_dim"] ** -0.5)
        nn.init.constant_(self.dec.trg_embedding.weight[0], 0)
                # nmtpytorch
                # with torch.no_grad():
                #     enc.src_embedding.weight.data[0].fill_(0)

        


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
            'enc_bidirectional': False,  # Whether the encoder is bidirectional or unidirectional.
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

    def _create_src_encoder(self):
        """
        Returns a transformer encoder.
        :return: a transformer encoder.
        """
        return TFEncoder(
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
    
    def _create_image_encoder(self):
        return super().create_image_encoder()

    def _create_decoder(self):
        """
        Returns a transformer decoder.
        :return: a transformer decoder.
        """
        return TFDecoder(
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
    
    def _create_cor_decoder(self):
        """
        Returns a transformer decoder.
        :return: a transformer decoder.
        """
        return TFDecoder(
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

    def get_attention_weights(self):
        return {
            'encoder_src': self.encoders['src'].get_attention_weights(),
            'decoder': self.dec.get_attention_weights()
        }

    def cache_enc_states(self, batch, **kwargs):
        if self.opts.model["enc_bidirectional"]:
            # It is tricky to cache the encoder's states if it's bidirectional,
            # as they are dependent on future positions due to the
            # self-attention module. Therefore, we are just going to cache the data
            # and perform the forward pass in get_enc_state_dict.
            self.current_batch = batch
        else:
            for key, enc in self.encoders.items():
                _ = enc(batch[key])

    def get_enc_state_dict(self, up_to=int(1e6)):
        """Encodes the batch optionally by partial encoding up to `up_to`
        words for derived simultaneous NMT classes. By default, the value
        is large enough to leave it as vanilla NMT. """
        if self.opts.model["enc_bidirectional"]:
            # In the bidirectional case, perform a forward pass through the encoder.
            return {str(key): encoder(self.current_batch[key][:up_to, :]) for key, encoder in self.encoders.items()}
        else:
            return super().get_enc_state_dict(up_to=up_to)

    def forward(self, batch, **kwargs):

        self.cache_enc_states(batch)
        encoded_src = self.get_enc_state_dict()

        # The input to the transformer should include the <bos> token but not the <eos> token.
        target_input = batch[self.tl][:-1, :]

        # The actual values should not have the <bos> token but should include the <eos>
        target_real = batch[self.tl][1:, :]

        cor_input = batch['src_cor'][:-1, :]
        cor_real = batch['src_cor'][1:, :]

        result, _ = self.dec(encoded_src, target_input, **kwargs)
        result_cor, _ = self.dec_cor(encoded_src, cor_input, **kwargs)

        mt_loss = self.loss(
            result.contiguous().view(-1, result.size(-1)), target_real.contiguous().view(-1))

        cor_loss = self.cor_loss(
            result_cor.contiguous().view(-1, result_cor.size(-1)), cor_real.contiguous().view(-1))

        cor_loss = cor_loss * self.sigma

        mt_dict = {
            'loss': mt_loss,
            'n_items': target_real.nonzero(as_tuple=False).size(0),
        }

        cor_dict = {
            'loss': cor_loss,
            'n_items': cor_real.nonzero(as_tuple=False).size(0)
        }

        return {
            'MT': mt_dict,
            'COR': cor_dict,
        }

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in progress_bar(data_loader, unit='batch'):
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

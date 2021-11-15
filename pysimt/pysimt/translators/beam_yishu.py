import logging
import time, operator
import torch
from queue import PriorityQueue
from ..utils.device import DEVICE
from ..utils.io import progress_bar
from ..utils.data import sort_predictions
# from torchtext.data.metrics import bleu_score
logger = logging.getLogger('nmtpytorch')


"""SimultaneousBeamSearch."""


class SimultaneousTF2TFBeamSearch:
    def __init__(self, model, data_loader, out_prefix, batch_size, filter_chain=None,
                 beam_width=12, max_len=100, model_opts=None, **kwargs):

        # logger.info(f'Ignoring batch_size {batch_size} for simultaneous beam search')
        # batch_size = 1

        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.filter_chain = filter_chain
        self.out_prefix = out_prefix

        self.vocab = self.model.trg_vocab
        self.n_vocab = len(self.vocab)
        self.unk = self.vocab['<unk>']
        self.eos = self.vocab['<eos>']
        self.bos = self.vocab['<bos>']

        self.max_len = max_len
        self.do_dump = out_prefix != ''

        self.beam_width = beam_width

        self.ratio = model.ratio
        self.lamb = model.lamb
        self.offset = model.offset

        # print(self.ratio, self.lamb, self.offset)

        self.buffer = None
        self.t_ptr = 0

        # print(kwargs)
        # assert(False)

    def dump_results(self, hyps, suffix=''):
        suffix = 'beam' if not suffix else f'{suffix}.beam'
        suffix += str(self.beam_width)

        # Dump raw ones (BPE/SPM etc.)
        self.dump_lines(hyps, suffix + '.raw')
        if self.filter_chain is not None:
            self.dump_lines(self.filter_chain(hyps), suffix)

    def dump_lines(self, lines, suffix):
        fname = f'{self.out_prefix}.{suffix}'
        with open(fname, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')

    def decoder_step(self, state_dict, next_word_idxs, h, hypothesis=None):
        logp, h = self.model.dec.f_next(
            state_dict, self.model.dec.get_emb(next_word_idxs), h, hypothesis)

        # Compute most likely word idxs
        next_word_idxs = logp.argmax(dim=-1)
        return logp, h, next_word_idxs

    def decoder_init(self, state_dict=None):
        return self.model.dec.f_init(state_dict)


    def clear_states(self):
        self.s_ptr = 0
        self.t_ptr = 0

        # self.prev_h = None
        # self._c_states = None
        # self.prev_word = None
        # self.eos_written = False

        self.actions = []

        if self.buffer is None:
            # Write buffer
            self.buffer = torch.zeros((100*self.max_len, ), dtype=torch.long, device=DEVICE)
        else:
            # Reset hypothesis buffer
            self.buffer.zero_()

    # def write(self, new_words):
    #     """Write the new words, move the pointer and accept the hidden state."""
    #     if new_words == None or len(new_words) == 0:
    #         return
    #     # print('new_words: ', new_words)
    #     # new_words = torch.cat(new_words, 1).reshape(-1)
    #     new_words = torch.Tensor(new_words, 1).reshape(-1)
    #     # print('new_words: ', new_words)
    #     length = new_words.shape[0]
    #     # print(length)
    #     # print(self.buffer[self.t_ptr:self.t_ptr+length])
    #     # print(new_words)
    #     # print('self.buffer: ', self.buffer)
    #     self.buffer[self.t_ptr:self.t_ptr+length] = new_words
    #     # print('self.buffer after: ', self.buffer)
    #     self.t_ptr += length

    # @profile
    def beam_search_old(self, Xi, Mi, yj_prev, hidden_prev, beam_width, max_len, y_prev_full): #, i, j):

        # self.clear_states()
        endnodes = []
        # print('new beam_search yj_prev: ', yj_prev.tolist(), self.vocab.idxs_to_sent(yj_prev.tolist()[0]))
        node = BeamSearchNode(hidden_prev, None, yj_prev, 0, 1, y_prev_full)
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))

        # print('#1')
        while True:
            # print('\n\nNew Interate: ', nodes.qsize())
            # print('endnodes: ', endnodes)
            # for (score, n) in nodes.queue:
                # print(n.wid.tolist()[0], self.vocab.idxs_to_sent(n.wid.tolist()[0]))
            # if node.yj_prev ==None:
                # print('node.yj_prev: None')
            # else:
                # print('node.yj_prev: {} {}'.format(node.yj_prev.tolist(), self.vocab.idxs_to_sent(node.yj_prev.tolist()[0]))    )
            # print('node.len: ', node.len)
            # queue is empty
            if nodes.qsize() == 0: 
                # print("Break: nodes.qsize() == 0")
                break
            # print('#1.1')
            (score, node) = nodes.get()

            # print('node.len new: ', node.len)
            # reach maximum len
            if node.len > self.max_len: 
                # print("Break: node.len {} > max_len {}".format(node.len, self.max_len))
                endnodes.append((score, node))
                break

            # decode EOS(id==2)
            if node.wid == self.eos and node.prev_node != None:
                endnodes.append((score, node))
                if len(endnodes) >= beam_width:
                    # print("Break: len(endnodes) >= beam_width")
                    break
                else:
                    continue

            # if node.shift == True:
                # continue

            # print('#1.2')
            # (0) RNN step
            yj_prev = node.wid
            h_prev = node.h
            y_prev_full = node.y_prev_full

            y_emb = self.model.dec.emb(yj_prev).reshape(1, -1)
            # print('y_emb: ', y_emb.shape)
            # assert(False)

            # logp_yj, prob_a_ij, h_curr = self.model.dec.f_get_single_step(Xi, Mi, y_emb, h_prev, i, j+node.len)

            logp_yj, prob_a_ij, h_curr = self.model.dec.forward_single(Xi, Mi, y_emb, h_prev, y_prev_full)

            # print('#1.3')
            # (1) SHIFT
            # logp_a_ij = prob_a_ij.log()
            prob_a_ij_neg = prob_a_ij
            prob_a_ij = 1-prob_a_ij_neg 
            logp_a_ij = prob_a_ij.log()

            # idx_x = i
            # idx_y = j+node.len
            # prior = (-self.model.lamb * torch.abs(idx_x - idx_y*self.model.ratio - self.model.offset))


            # print('Shift/Emit: ', prob_a_ij.tolist())
            if prob_a_ij > torch.Tensor([0.5]).to(DEVICE):
                # print('SHIFT==')
                node.logp = node.logp + logp_a_ij
                node.shift = True
                node.h_prev = h_prev
                node.yj_prev = yj_prev
                endnodes.append((-node.eval(), node))
                if len(endnodes) >= beam_width:
                    # print('Break: Reach max beam')
                    break
                else:
                    continue
                
            # print('#1.4')
            # (2) EMIT 
            # print('EMIT==')
            node.shift = False
            logp_yj_topk, indices = torch.topk(logp_yj, beam_width)

            # print('==')
            # print(logp_yj_topk, indices)

            for new_k in range(0, beam_width):
                wid = indices[0][new_k].view(1, -1)
                # print('new_k:', new_k)
                log_y = logp_yj_topk[0][new_k].item()
                # print('indices: ', indices.tolist())
                # print('wid: ', wid.tolist())
                y_prev_full = node.y_prev_full + wid.data.tolist()[0]

                # print('y_prev_full:', y_prev_full)

                node = BeamSearchNode(h_curr, node, wid, node.logp + log_y + torch.log(1-prob_a_ij), node.len + 1, y_prev_full)
                node.yj_prev = yj_prev
                # node.y_prev_full.append(wid) # YM: track y
                nodes.put((-node.eval(), node))

        # print('#2')
        # if len(endnodes) == 0:
        #     endnodes = [nodes.get()]

        # print('#3')
        _, node = sorted(endnodes, key=operator.itemgetter(0))[0] # get the top node

        

        utterance = []
        if node.shift == True: # roll back RNN step
            # hidden = node.h_prev
            # yj = node.yj_prev   # dont append yj

            if node.prev_node == None:  # 1) no decoded words, return result
                hidden = node.h_prev
                yj = node.yj_prev 
                return utterance, hidden, yj
            else:
                node = node.prev_node   # 2) go back to the emitted words

        hidden = node.h
        yj = node.wid
        
        # print("Top node y_prev_full:", node.y_prev_full)
        # print("Top node wid:", node.wid)

        while node != None:   # get all the previous words
            if node.shift == False:
                # print('Loop node.wid: {}, yj: {}'.format(node.wid.tolist(), yj))
                if node.prev_node != None:
                    utterance.insert(0, node.wid)  # there is only one shift
                # print("Utterance:", utterance)
            # else:
            #     print("shift???")
            #     assert(False)
            node = node.prev_node

        return utterance, hidden, yj


    def complete_beam(self, beam_ids, beam_parent, beam_logits, beam_hidden, id_parent):
        # beam_ids = [ 1 2 2 ]
        #              9 3 1
        #              3 8 6
        # beam_parent = [ 0 0 2 ]
        #                 0 2 0
        #                 0 1 1
        # id_parent [2] -> 1, 2, [0]
        # path             3, 3, 6
        # print(" complete_beam IN :", beam_ids[0].shape, beam_parent[0].shape, beam_logits[0].shape, beam_hidden[0].shape, id_parent)
        path = []

        logit = beam_logits[-1][id_parent]
        # hidden =  beam_hidden[-1][id_parent].reshape(1,1,-1)

        for i in range(len(beam_logits)):
            idx = beam_ids[-i-1][id_parent].data.tolist()
            
            id_parent = beam_parent[-i-1][id_parent].data.tolist()

            path.insert(0, idx)

        return logit, path, None

    def beam_search(self, Xi, Mi, yj_prev, hidden_prev, beam_width, max_len, y_prev_full, i_pos, j_pos, prob_b_cum_renorm, ratio_label):
        
        # ------
        # step 0:
        # ------
        # y_emb = self.model.dec.emb(yj_prev).reshape(1, -1)
        # print(Xi.shape, Mi.shape, y_emb.shape, hidden_prev.shape)
        # if hidden_prev!=None:
            # print('\n###-hidden_prev:', hidden_prev.shape)
            # print(Xi.shape, Mi.shape, y_emb.shape, hidden_prev.shape)
        # else:
            # print('\n###-hidden_prev:', hidden_prev)
            # print(Xi.shape, Mi.shape, y_emb.shape)
    

        y_prev_full = torch.tensor(y_prev_full).long().cuda().reshape(-1, 1)    # [n, b=1]
        logp_yj, logp_a, num_x, num_y, hidden = self.model.dec.forward_single(Xi, Mi, None, None, y_prev_full, i_pos, j_pos)

        # -------- breaking probability------------
        m = i_pos + self.max_len
        idx_x = torch.arange(0, m, device=DEVICE).unsqueeze(0).view(1, m) # b, m
        
        # ratio = (num_x + i_pos)/(num_y + j_pos + 1e-8) # b
        # ratio = ratio.unsqueeze(1)  # b, 1
        # ratio = self.ratio
        ratio = ratio_label
        # print(ratio, ratio_label)
        prior = self.lamb * (idx_x - j_pos*ratio - self.offset) * (idx_x - j_pos*ratio - self.offset)

        mask_buffer = (idx_x > j_pos*ratio + self.offset).float()

        prob_b = torch.sigmoid(prior)*mask_buffer
        prob_b_select = prob_b[:,i_pos].view(-1, 1)

        # -------- breaking probability------------
        # print(ratio.shape, logits.shape, logits_sum.shape)
        prob_a = logp_a[:,1].exp()
        

        prob_emit_step = prob_a + (1-prob_a)*prob_b_select
        # prob_emit_step = prob_a

        # print(i_pos, j_pos, ratio.data.tolist(), prob_a.data.tolist(), prob_b_select.data.tolist(), prob_emit_step.data.tolist())

        

        logp_emit_step = prob_emit_step.log()
        prob_shift_step = 1-prob_emit_step
        logp_shift_step = prob_shift_step.log()

        # ---------
        # 0) SHIFT, read source words
        if prob_shift_step > torch.Tensor([0.5]).to(DEVICE):
        # if logp_shift_step > logp_emit_step: #+ logp_yj_topk[0][0]:
            # print('=SHIFT')
            return [], None, yj_prev

        # ---------
        # 1) EMIT, generate target words
        # print(logp_yj.shape, prob_a_ij.shape)
        logp_sum = logp_yj + logp_emit_step
        logp_yj_topk, indices = torch.topk(logp_sum, beam_width)

        # print(indices.shape, hidden.shape) # 
        ids = indices # 1, 10 | batch, beam_width
        # hidden = hidden.expand(1, beam_width, -1).contiguous()  # 1, 1, 512 | len, batch, hsize -> 1, 10, 512 | len, beam_width, hsize

        # print(hidden.shape)
        

        # print(Xi.shape, Mi.shape)
        Xi = Xi.expand(-1, beam_width, -1) # m, b, h
        Mi = Mi.expand(beam_width, 1,  -1) # b, 1, m

        log_shift_max = -1e30

        shift_logits = [logp_shift_step]
        # shift_hidden_prev = hidden_prev
        shift_y = yj_prev
        shift_path = []
        
        emit_ids = [ids[0]]                         # (1), 10
        emit_parent = [torch.zeros_like(ids[0])]    # (1), 10
        emit_logits = [logp_yj_topk[0]]             # (1), 10 
        # emit_hidden = [hidden.squeeze(0)]           # (1), 10, 512


        y_prev_full = y_prev_full.expand(-1, beam_width) # n, beam_width(batch)
        y_prev_full = torch.cat([y_prev_full, ids], dim=0)
        # emit_path_record = y_prev_full   # [n, b]


        curr_len = 1

        while True:
            if curr_len > max_len:
                break

            # ----------
            # Model Step
            # y_emb = self.model.dec.emb(ids).squeeze(0) # 1, 10, 512 | len, beam_width, hsize
            # print(y_emb.shape)
            # print(hidden.shape)
            # print(y_prev_full)
            logp_yj, logp_a, num_x, num_y, _ = self.model.dec.forward_single(Xi, Mi, None, None, y_prev_full, i_pos, j_pos + curr_len)
            # 10, 6435 | 10, 1 | 1, 10, 512
            # print(logp_yj.shape, prob_a_ij.shape, hidden.shape)

            # -------- breaking probability------------
            m = i_pos + self.max_len
            idx_x = torch.arange(0, m, device=DEVICE).unsqueeze(0).view(1, m) # b, m
            
            # ratio = (num_x + i_pos)/(num_y + j_pos + 1e-8) # b
            # ratio = ratio.unsqueeze(1)  # b, 1
            ratio = ratio_label
            # ratio = self.ratio
            # print(ratio, ratio_label)
            prior = self.lamb * (idx_x - j_pos*ratio - self.offset) * (idx_x - j_pos*ratio - self.offset)

            mask_buffer = (idx_x > j_pos*ratio + self.offset).float()

            prob_b = torch.sigmoid(prior) * mask_buffer
            prob_b_select = prob_b[:,i_pos].view(-1, 1)


            # ----------------------------------------
            # Beam Search
            prob_a = logp_a[:,1].exp().unsqueeze(1)  
            prob_emit_step = prob_a + (1-prob_a)*prob_b_select  
            # prob_emit_step = prob_a
            # print('prob_emit_step', prob_emit_step)
            logp_emit_step = prob_emit_step.log()
            prob_shift_step = 1-prob_emit_step
            logp_shift_step = prob_shift_step.log()

            # ----------------------------------------
            # 1) SHIFT
            # log_shift = logp_shift_step.reshape(1, beam_width) + emit_logits[-1] # shift + history, # 1, 10
            # print(logp_shift_step.shape, emit_logits[-1].shape)
            log_shift = logp_shift_step.reshape(1, beam_width) + emit_logits[-1]/(curr_len+1)
            shift_logits.append(log_shift)
            log_shift_curr, id_shift_parent = torch.max(log_shift, dim=1)
            id_shift_parent = id_shift_parent.data.tolist()[0]

            # YM: Lenght normalisation
            # log_shift_curr_renorm = log_shift_curr/(curr_len+1)
            # log_shift_curr_renorm = log_shift_curr

            if  log_shift_curr > log_shift_max:
                log_emit_prev, shift_path, _ = self.complete_beam(emit_ids, emit_parent, emit_logits, None, id_shift_parent)
                # print(shift_path)
                log_shift_max = log_shift_curr
                # print(" log_shift_curr > log_shift_max: ", shift_hidden_prev.shape)

            # ----------------------------------------
            # 2) EMIT
            # print('--------------')
            # print(logp_emit_step)
            # print(logp_yj)
            # print(emit_logits)
            log_emit = logp_emit_step + logp_yj + emit_logits[-1].reshape(beam_width, 1)  # emit + word + history, # 10, 6435
            log_emit = log_emit.reshape(1, -1) # beam_width*voc
            logp_topk, indices = torch.topk(log_emit, beam_width) # 1,  6435
            # print('--------------')
            # print(indices)
            # print(logp_topk)

            #  | 1, 20
            # print(logp_yj_topk.shape, indices.shape)
            ids = indices % self.n_vocab # Which word in vocabulary.
            parent = indices // self.n_vocab # Which hypothesis it came from.

            emit_ids.append(ids[0])
            emit_logits.append(logp_topk[0])
            emit_parent.append(parent[0])

            # print('--------------')
            # y_prev_full = torch.arange(0, 10, device=DEVICE).unsqueeze(0)
            # print(y_prev_full) # [n, b]
            

            n_curr, _ = y_prev_full.shape

            # print(parent[0], ids[0])

            # idx_i = torch.arange(0, n_curr, device=DEVICE)
            # idx_new = torch.concat([idx_i, parent[0]])

            idx_new = parent.expand(n_curr, -1)

            # print('--------------')
            # print(indices,  self.n_vocab, indices // self.n_vocab)
            # print(y_prev_full, parent, idx_new)

            y_prev_full = y_prev_full.gather(1, idx_new)

            y_prev_full = torch.cat([y_prev_full, ids], dim=0) # [n, b], [1, b]

            # print(y_prev_full)

            # emit_hidden.append(hidden[0])
            # new_hidden = torch.stack([hidden[0, pid,:]  for pid in parent[0]], dim=0)
            # emit_hidden.append(new_hidden) # 10, 512
            
            curr_len +=1

        log_emit_curr = emit_logits[-1][0]
        # print(emit_parent)
        # print(emit_parent[-1])
        # print(emit_parent[-1][0])
        # print(emit_parent[-1][0].data.tolist())
        id_emit_parent = emit_parent[-1][0].data.tolist()
        
        log_emit_prev, emit_path, _ = self.complete_beam(emit_ids[:-1], emit_parent[:-1], emit_logits[:-1], None, id_emit_parent)
        log_emit_max = log_emit_curr
        # emit_hidden_max = emit_hidden[-1][0].reshape(1,1,-1)



        # print('---')
        # print(emit_hidden_prev.shape)
        # if shift_hidden_prev !=None:
            # print(shift_hidden_prev.shape)

        if log_emit_max > log_shift_max:
            # print('==EMIT END')
            if len(emit_path) > 0:
                # print('emit???')
                emit_y = torch.tensor(emit_path[-1]).to(DEVICE).reshape(1,1)
            else:
                # print('emit???')
                emit_y = yj_prev
            return emit_path, None, emit_y
        else:
            # print('==SHIFT END')
            if len(shift_path) > 0:
                # print('shift 00???')
                # print(shift_path[-1])
                shift_y = torch.tensor(shift_path[-1]).to(DEVICE).reshape(1,1)
            else:
                # print('shift 11???')
                shift_y = yj_prev
                # print('shift_hidden_prev:',shift_hidden_prev)
            return shift_path, None, shift_y

    def run_all(self):
        return self.run()

    # @profile
    def run(self, **kwargs):

        actions = []

        translations = []
        references = []

        candidates = []

        start = time.time()

        count = 0 

        AL_score = []

        for batch in progress_bar(self.data_loader, unit='batch'):
            # print(batch)
            # assert(False)

            self.clear_states()

            batch.device(DEVICE)

            # Cache encoder states
            self.model.cache_enc_states(batch)

            # Get encoder hidden states
            state_dict = self.model.get_enc_state_dict()

            # Initial state is None i.e. 0. state_dict is not used
            # h = self.decoder_init(state_dict)



            encoder_outputs = []
            encoder_masks = []
            for k, (src, mask) in state_dict.items():
                encoder_outputs.append(src)
                encoder_masks.append(mask)

            x_len = torch.sum(encoder_masks[0] ==False, (1, 2))
            y_len = [batch[self.model.tl].gt(0).sum(0)][0] -1  # -1?
            ratio_label = x_len*1.0/y_len
            # print(ratio_label)

            # print(encoder_outputs)
            # print(encoder_masks)
            
            m = encoder_outputs[0].shape[0]

            #---------------------------Diagonal--------------
            # buffer size prob
            # self.ratio = 1
            # self.lamb = 3.0
            # self.offset = 1.0
            # n = 1000
            # idx_x = torch.arange(0, m, device=DEVICE).unsqueeze(1).expand(m, n).view(1, m, n)
            # idx_y = torch.arange(0, n, device=DEVICE).unsqueeze(0).expand(m, n).view(1, m, n)
            # prior = (-self.lamb * torch.abs(idx_x - idx_y*self.ratio - self.offset))

            # mask_buffer = (idx_x > idx_y*self.ratio + self.offset).float() + 1e-7
            # mask_buffer[:,-1,:] = 1.0

            # prob_b = torch.softmax(prior, dim=1)
            # prob_b = prob_b * mask_buffer
            # prob_b = prob_b / prob_b.sum(dim=1, keepdim=True)

            # log_prob_b = prob_b.log()
            # prob_cum = torch.flip(torch.cumsum(torch.flip(prob_b, [1]), dim=1), [1]) # b, m, n
            # prob_b_cum_renorm = prob_b/(prob_cum + 1e-8)
            prob_b_cum_renorm = None
            #---------------------------Diagonal--------------

            # AL
            g_t = []
            y_prev_full_curr = [1]

            for i in range(0, m):
                # print('\nX index: {}'.format(i))

                # print(self.model)
                # print('\nencoder_outputs:', encoder_outputs[0].shape)
                # print('encoder_masks:', encoder_masks[0].shape)

                Xi = encoder_outputs[0][:i+1, :, :]     #.view(1,-1)  # [s, 1, h]
                Mi = encoder_masks[0][:, :, :i+1]


                # full_mask = (encoder_masks[0].squeeze(1).t()==False).long()
                # Mi = full_mask[:i+1, :]

                # print(Mi, Mi.shape)
                # assert(False)

                # YM: Transformer
                # Mi = encoder_masks[0][:,:, :i+1]

                # print('Xi Mi:')
                # print(Xi.shape, Mi)

                # if i==0:
                #     hidden_prev = None
                #     yj_prev = self.model.get_bos(self.batch_size).to(DEVICE).view(1,-1)

                # print(y_prev_full_curr)
                i_pos = i
                j_pos = len(y_prev_full_curr) -1

                utterances, hidden, yj = self.beam_search(Xi, Mi, None, None, self.beam_width, self.max_len, y_prev_full_curr, i_pos, j_pos, prob_b_cum_renorm, ratio_label)
              
                # print('==')
                # if hidden != None:
                    # print(utterances, hidden.shape, yj)
                # assert(False)
                # print('==')
                # yj_prev = yj
                # hidden_prev = hidden

                # self.write(utterances)
                # print(utterances)
                for idx, word in enumerate(utterances):
                    if word == self.eos:
                        utterances = utterances[:idx+1]
                        break
                # print(i, utterances)
                # print(utterances == None, len(utterances) == 0)

                if utterances == None or len(utterances) == 0:
                    # print('y_prev_full_curr IN:', y_prev_full_curr)
                    y_prev_full_curr = y_prev_full_curr
                else:
                    y_prev_full_curr += utterances

                g_t += [i+1 for _ in range(0, len(utterances))]
                
                if yj == self.eos:
                    # print("End early")
                    break

            # All finished, convert translations to python lists on CPU
            # idxs = self.buffer[self.buffer.ne(0)].tolist()
            idxs =  y_prev_full_curr[1:]
            # print("idxs: ", idxs)
            if len(idxs) ==0 or idxs[-1] != self.eos:
                # In cases where <eos> not produced and the above loop
                # went on until max_len, add an explicit <eos> for correctness
                idxs.append(self.eos)
                g_t.append(m)

            # print("idxs add eos: ", idxs)
            candidate = self.vocab.idxs_to_sent(idxs)

            y = batch[self.model.tl][:,0].tolist()
            # print('\nidxs: ', idxs)
            # print('y: ', y)
            referece = self.vocab.idxs_to_sent(y[1:])
            # print('\nreferece:  ', referece)
            # print('candidate: ', candidate)
            if len(candidate) ==0:
                candidate = '.'
            candidates.append(candidate)

            words = candidate.split()

            translations.append(words)
            references.append([referece.split()])


            ratio =  len(words) / m # 

            # print('ratio: ', ratio)
            # print('g_t: ', g_t)

            AL_sum = []
            cut_off = 0
            for t, val in enumerate(g_t):
                AL_sum.append(val-(t)/ratio)  
                if val == m: # cut-off
                    cut_off = t+1
                    break
            if cut_off==0:
                cut_off = g_t[-1]

            AL_score.append(sum(AL_sum)/cut_off)


            # print('cut-off: ', cut_off)
            # print('AL_score: ', AL_score)

            # count += 1
            # if count >20:
            #     break
            # break
            
        
        # print('translations: ' , translations)
        hyps = sort_predictions(self.data_loader, candidates)
        # print('hyps: ' , translations)
        up_time = time.time() - start

        # print(translations, references)
        # bscore = bleu_score(translations, references)
        # logger.info("Test BLEU: {} | Test AL: {}".format(bscore, sum(AL_score)/len(AL_score)))

        if self.do_dump:
            self.dump_results(hyps)

        return (hyps, actions, up_time)


class BeamSearchNode(object):
    def __init__(self, hidden, prev_node, wid, logp, length, y_prev_full, STA=False):
        '''
        :param hidden:
        :param prev_node:
        :param wid:
        :param logp:
        :param len:
        '''
        self.h = hidden
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.len = length
        self.shift = False
        self.h_prev = None
        self.yj_prev = None
        self.STA = STA

        self.y_prev_full = y_prev_full

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp #/ float(self.len - 1 + 1e-6) + alpha * reward

    def __gt__(self, other):
        return -self.eval() > -other.eval()

    def __eq__(self, other):
        if other == None:
            return False
        else:
            return -self.eval() == -other.eval()

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, src_dict, dst_dict):
        super().__init__(args, src_dict, dst_dict)
        self.eps = args.label_smoothing
        self.copy_mechanism = args.copy_mechanism
        
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        toprint = False
        
        net_output, avg_attn_scores, final_pgen_scores = model(**sample['net_input'])
        # print("net_output: ", net_output.size())
        # print("net_output: ", net_output)
        # print("avg_attn_scores: ", avg_attn_scores)
        if toprint: print("final_pgen_scores: ", final_pgen_scores.size())
        if toprint: print(final_pgen_scores[0])
        if toprint: print(sample["copy_mechanism"]["src_oov_words"][0])
        if toprint: print(sample["net_input"]["src_tokens"][0])
        
        if self.copy_mechanism:
            batch_size = net_output.size()[0]
            batch_dec_len = net_output.size()[1]
            dest_vocab_size = net_output.size()[2]
            
            # Original Vocab Distribution
            vocab_dist = F.softmax(net_output, dim=2)
            if toprint: print("vocab_dist", vocab_dist[0][0], sum(vocab_dist[0][0]))
            vocab_dist = final_pgen_scores * vocab_dist
            # print("vocab_dist: ", vocab_dist.size())
            # print(final_pgen_scores[0][0])
            if toprint: print("vocab_dist", vocab_dist[0][0], sum(vocab_dist[0][0]))

            if toprint: print("avg_attn_scores", avg_attn_scores[0][0], sum(avg_attn_scores[0][0]))
            
            # Original Attention Distribution: avg_attn_scores is already normalized, no softmax again.
            # atten_dist = F.softmax(avg_attn_scores, dim=2)
            # print(atten_dist[0][0], sum(atten_dist[0][0]))
            atten_dist = (1-final_pgen_scores) * avg_attn_scores # atten_dist
            # print("atten_dist: ", atten_dist.size())
            if toprint: print("atten_dist", atten_dist[0][0], sum(atten_dist[0][0]))            
            
            # Extra Zeros: ADD ONE TO ENSURE IT IS NOT 0
            max_src_oov_words = sample["copy_mechanism"]["max_src_oov_words"] + 1 
            extra_zeros = torch.zeros(batch_size, batch_dec_len, max_src_oov_words).cuda()
            # print("extra_zeros: ", extra_zeros.size())
            
            # Extended Vocab Distribution
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=2)
            if toprint: print("extended_vocab_dist: ", extended_vocab_dist.size())
            if toprint: print(extended_vocab_dist[0][0], sum(extended_vocab_dist[0][0]))

            
            # Repeat src_tokens_extended
            src_tokens_extended = sample["copy_mechanism"]["src_tokens_extended"]
            src_tokens_extended = src_tokens_extended.unsqueeze(1).repeat(1, batch_dec_len, 1)
            if toprint: print("src_tokens_extended: ", src_tokens_extended.size())
            if toprint: print(src_tokens_extended[0][0])
            
            # Prepare final dist
            final_dist = extended_vocab_dist.scatter_add_(2, src_tokens_extended, atten_dist)
            final_dist = torch.clamp(final_dist, min=1e-8) # Clamp to avoid zero prob for oov words, otherwise log of that will be 0
            if toprint: print("final_dist: ", final_dist.size())
            if toprint: print(final_dist[0][0], final_dist[0][0][50000:], sum(final_dist[0][0]))

            # Get Final Data: Clamp to avoid zero prob for oov words
            lprobs = torch.log(final_dist)            
            target = sample["copy_mechanism"]['target_extended'].unsqueeze(-1)
            if toprint: print("lprobs, target:", lprobs.size(), target.size())
            if toprint: print(target[0], lprobs[0][0], final_dist[0][0], sum(final_dist[0][0]))
            if toprint: print(lprobs[0][0][50000:], final_dist[0][0][50000:], sum(final_dist[0][0]))
            
            # exit(0)
            
            # print("TEST:")
            # lprobs = model.get_normalized_probs(net_output, log_probs=True)
            # target = sample['target'].unsqueeze(-1)
            # print(target[0], lprobs[0][0], F.softmax(net_output, dim=2)[0][0], sum(F.softmax(net_output, dim=2)[0][0]))
            # exit(0)
            
            # print("lprobs, target:", lprobs.size(), target.size())
            # print(target)
        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            target = sample['target'].unsqueeze(-1)
            
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
        
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
        }

from __future__ import print_function

import torch
import torch.nn as nn


class HACLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(HACLoss, self).__init__()
        self.temperature = temperature # hyperparameters
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def check_positives(self, h_mask):
        temp_sum = h_mask.sum(1)
        flag = True
        for i in temp_sum:
            if i == 0.:
                flag = False
                break           
        return flag

    def forward(self, features, labels=None, mask=None, w_super=None, hierarchy = [[3, 5], [0, 1, 4, 2]]):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        # not good coding style
        elif self.contrast_mode == 'hil':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            hie_labels = []
            for i in labels:
                for j in i:
                    for h_i, h_d in enumerate(hierarchy):
                        if j in h_d:
                            hie_labels.append([h_i])
            hie_labels = torch.Tensor(hie_labels)
            h_mask = torch.eq(hie_labels, hie_labels.T).float().to(device)
            mask = (h_mask - mask) * w_super + mask
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)        
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        mask = mask.repeat(anchor_count, contrast_count)        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)       
        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()   
        return loss

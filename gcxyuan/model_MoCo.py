import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class MoCo(nn.Module):

    def __init__(self, base_encoder, config):
        super(MoCo, self).__init__()
        
        #encoder
        self.encoder_q = base_encoder(input_dim = config['input_dim'], num_class=config['num_class'],low_dim=config['low_dim'])
        #momentum encoder
        self.encoder_k = base_encoder(input_dim = config['input_dim'], num_class=config['num_class'],low_dim=config['low_dim'])
        #decoder
        

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(config['low_dim'], config['moco_queue']))        
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(config['num_class'],config['low_dim']))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, config):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * config['moco_m'] + param_q.data * (1. - config['moco_m'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, config):
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert config['moco_queue'] % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % config['moco_queue']  # move pointer

        self.queue_ptr[0] = ptr
    
    def forward(self, img, target, config, is_eval=False, is_proto=False):

        #img = batch[0].cuda(non_blocking=True)        
        #target = batch[1].cuda(non_blocking=True) 
        
        output, q, u, x_q = self.encoder_q(img)
        if is_eval:  
            return output, q, target
            
        #img_aug = batch[2].cuda(non_blocking=True)
        img_aug = img.clone().cuda()
        # compute augmented features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(config)  # update the momentum encoder
            
            _, k, _, _ = self.encoder_k(img_aug)  
            
        # compute instance logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= config['temperature']
        #inst_labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, config) 
        
        if is_proto:     
            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_proto = torch.mm(q,prototypes.t())/config['temperature']        
        else:
            logits_proto = 0
         
        targets = target
        features = q
        
        # update momentum prototypes with original labels
        for feat,label in zip(features,targets):
            self.prototypes[label] = self.prototypes[label]*config['proto_m'] + (1-config['proto_m'])*feat            

        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)        

        return output, target, logits, x_q, logits_proto, u


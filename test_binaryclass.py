#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-10-29 20:36

import torch
from core.solver import BaseSolver

from core.asr import ASR, Encoder_Classifier
from core.optim import Optimizer
from core.data import load_biclass_dataset
from core.util import human_format, cal_er, feat_to_fig


class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_wer = {'att': 3.0, 'ctc': 3.0}
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']
        
#     def load_my_state_dict(self, state_dict):
 
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state:
#                  continue
#             if isinstance(param, Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             own_state[name].copy_(param)
       
    def load_ckpt(self):
        """
         Load ckpt if --load option is specified
        :return:
        """
        if self.paras.load:
            # Load weights
            ckpt = torch.load(self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
#             pretrained_dict = {k: v for k, v in ckpt.items() if k in self.model.state_dict()}
#             self.model.state_dict().update(pretrained_dict) 
#             self.model.load_state_dict(self.model.state_dict())

           
                
            self.model.load_state_dict(ckpt['model'])
            

            if self.emb_decoder is not None:
                self.emb_decoder.load_state_dict(ckpt['emb_decoder'])
            # if self.amp:
            #    amp.load_state_dict(ckpt['amp'])
            # Load task-dependent items
            for k, v in ckpt.items():
                if type(v) is float:
                    metric, score = k, v
            if self.mode == 'train':
                self.step = ckpt['global_step']
#                 self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                self.verbose('Load ckpt from {}, restarting at step {} (recorded {} = {:.2f} %)'.format(
                    self.paras.load, self.step, metric, score))
            else:
                self.model.eval()
                if self.emb_decoder is not None:
                    self.emb_decoder.eval()
#                 self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(
#                     self.paras.load, metric, score))

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return feat, feat_len, txt, txt_len

    def load_data(self):
        print("Load data for training/validation, store tokenizer and input/output shape")
        self.dv_set, self.tt_set, self.feat_dim, msg = \
            load_biclass_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        print("Setup ASR model and optimizer ")
        nonfreeze_keys = ['fc.weight', 'fc.bias']
        # Model
        self.model = Encoder_Classifier(self.feat_dim, **
        self.config['model']).to(self.device)
#         self.verbose(self.model.create_msg())
#         model_paras = [{'params': self.model.parameters()}]

#         print("# Losses")
#         self.bceloss = torch.nn.BCELoss()
#         print("# Note: zero_infinity=False is unstable?")
# #         self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)


#         print("# Optimizer")
#         self.optimizer = Optimizer(model_paras, **self.config['hparas'])
#         self.verbose(self.optimizer.create_msg())

#         print("# Enable AMP if needed")
#         self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
#         for name, para in self.model.named_parameters():
#             if para.requires_grad and name not in nonfreeze_keys:
#                 para.requires_grad = False
#         for name, para in self.model.named_parameters():
#             if para.requires_grad:print(name)
#         non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = Optimizer(non_frozen_parameters, **self.config['hparas'])

        # ToDo: other training methods

    def exec(self):
        ''' Testing End-to-end ASR system '''
        names=[]
        hyps=[]
        txts=[]
        ans=[]
        for data in self.tt_set:
            name, feat, feat_len, txt = data
            feat = feat.to(self.device)
            feat_len = feat_len.to(self.device)
            txt = txt.to(self.device)
            
            with torch.no_grad():
                hyp = self.model(feat, feat_len)
            an = ((hyp>=0.5) == (txt==1)).tolist()[0]
            print(name, ' ', hyp.tolist()[0], ' ', txt.tolist()[0], ' ', an)
            names.append(name[0])
            hyps.append(hyp.tolist()[0])
            txts.append(txt.tolist()[0])
            ans.append(((hyp>=0.5) == (txt==1)).tolist()[0])
            
            
        self.verbose('All done !')
        return names, hyps, txts, ans

    def validate(self):
        # Eval mode
        self.model.eval()
        dev_wer = {'att': [], 'ctc': []}

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                output = self.model(feat, feat_len)

            loss = self.bceloss(output, txt)

#             # Show some example on tensorboard
#             if i == len(self.dv_set) // 2:
#                 for i in range(min(len(txt), self.DEV_N_EXAMPLE)):
#                     if self.step == 1:
#                         self.write_log('true_text{}'.format(
#                             i), self.tokenizer.decode(txt[i].tolist()))
#                     if att_output is not None:
#                         self.write_log('att_align{}'.format(i), feat_to_fig(
#                             att_align[i, 0, :, :].cpu().detach()))
#                         self.write_log('att_text{}'.format(i), self.tokenizer.decode(
#                             att_output[i].argmax(dim=-1).tolist()))
#                     if ctc_output is not None:
#                         self.write_log('ctc_text{}'.format(i), self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
#                                                                                      ignore_repeat=True))

        # Ckpt if performance improves
        self.save_checkpoint('latest.pth', 'loss',
                             loss, show_msg=False)
        if loss < self.bestloss:
                self.bestloss = loss
                self.save_checkpoint('best_biclass.pth', 'loss', loss)
#         for task in ['att', 'ctc']:
#             dev_wer[task] = sum(dev_wer[task]) / len(dev_wer[task])
#             if dev_wer[task] < self.best_wer[task]:
#                 self.best_wer[task] = dev_wer[task]
#                 self.save_checkpoint('best_{}.pth'.format(
#                     task), 'wer', dev_wer[task])
#             self.write_log('wer', {'dv_' + task: dev_wer[task]})

        # Resume training
        self.model.train()
        if self.emb_decoder is not None:
            self.emb_decoder.train()

    def print_model(self):
        self.model = Encoder_Classifier(self.feat_dim,
                         **self.config['model'])
#         nonfreeze_keys = ['decoder.layers.weight_ih_l1','decoder.layers.weight_hh_l1', 'decoder.layers.bias_ih_l1', 'decoder.layers.bias_hh_l1']
        nonfreeze_keys = ['fc.weight', 'fc.bias']



        ckpt = torch.load(self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
#         pretrained_dict = {k: v for k, v in ckpt.items() if k in self.model.state_dict()}
#         self.model.state_dict().update(pretrained_dict) 
#         self.model.load_state_dict(self.model.state_dict())
           
                
#         self.model.load_state_dict(ckpt['model'])
        print(ckpt)
        print(self.model)
        for name, para in self.model.named_parameters():
            if para.requires_grad and name not in nonfreeze_keys:
                para.requires_grad = False
            print("-"*20)
            print(f"name: {name}")
            print("values: ")
            print(para)
        for name, para in self.model.named_parameters():
            if para.requires_grad:print(name)
        # Beam decoder
#         self.decoder = BeamDecoder(
#             self.model.cpu(), self.emb_decoder, **self.config['decode'])
#         self.verbose(self.decoder.create_msg())
        del self.model
#         del self.emb_decoder
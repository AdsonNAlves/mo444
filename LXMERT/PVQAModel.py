#!/usr/bin/env python
# coding: utf-8

from torch import nn

from src.lxrt.tokenization import BertTokenizer
from src.lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG

import torch.nn as nn

from src.parameters import args
from src.lxrt.entry import LXRTEncoder as LXRTEncoder_e
from src.lxrt.modeling import BertLayerNorm, GeLU

class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode=mode
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        output = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)


# Max length including <bos> and <eos>
MAX_PVQA_LENGTH = 20

class PVQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        # lxrt.entry.LXRTEncoder -> LXRTFeatureExtraction -> LXRTModel
        self.lxrt_encoder = LXRTEncoder_e(
            args,
            max_seq_length=MAX_PVQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit
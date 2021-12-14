"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import os
import base64
import json
import pickle
import numpy as np
import pandas as pd
import utils
import h5py
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
import itertools
import math

import re
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

# TODO: merge dataset_cp_v2.py with dataset.py

COUNTING_ONLY = False
question_types = ('where', 'what', 'how', 'how many/how much', 'when', 'why', 'who/whose', 'other', 'yes/no')
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def load_tsv(split: str, args = ''):
    tsv_file = 'data/pvqa/images/%s%s.csv' % (split, args)
    df = pd.read_csv(tsv_file, delimiter='\t', names=FIELDNAMES)

    data = []
    for i in range(df.shape[0]):
        datum = {}
        datum['img_id'] = '%s_%04d' % (split, df['image_id'][i])
        datum['img_w'] = df['image_w'][i]
        datum['img_h'] = df['image_h'][i]
        datum['num_boxes'] = df['num_boxes'][i]

        boxes = df['boxes'][i]
        buf = base64.b64decode(boxes[1:])
        temp = np.frombuffer(buf, dtype=np.float64).astype(np.float32)
        datum['boxes'] = temp.reshape(datum['num_boxes'], -1)

        features = df['features'][i]
        buf = base64.b64decode(features[1:])
        temp = np.frombuffer(buf, dtype=np.float32)
        datum['features'] = temp.reshape(datum['num_boxes'], -1)

        data.append(datum)

    return data

def is_ans_valid(ans):
    if ans in ('yes', 'no'):
        return 0
    else:
        return 1

def normalize_bbox(im_w, im_h, bbox):
    bbox = bbox.copy()
    bbox[:, 0] /= im_w
    bbox[:, 1] /= im_h
    bbox[:, 2] /= im_w
    bbox[:, 3] /= im_h
    return bbox 

def _load_dataset_pvqa(dataroot, name, imd_id2val, label2ans, ans2label):
    vqa = pickle.load(open(os.path.join(dataroot, 'qas/%s_vqa.pkl' % name), 'rb'))
    """
    'answer_type': 'other',
  'img_id': 'train_0422',
  'label': {'in the canals of hering': 1},
  'question_id': 100422000,
  'question_type': 'where',
  'sent': 'Where are liver stem cells (oval cells) located?'},
    """
    entries = []
    for qa in vqa:
        # print('qa')
        # print(qa)
        answer = None
        if True:  # name != 'test':s
            labels = list(qa['label'].items())
            """
            'labels': [2225, 124],
            'scores': [1.0, 0.3]
            """

            answer = {'labels': [ans2label[label[0]] for label in labels if label[0] in ans2label],
                      'scores': [label[1] for label in labels if label[0] in ans2label]}
            # print('answer: ', answer)

        ans = list(qa['label'].keys())[0]
        # if ans is None or ans == 'yes' or ans == 'no':
        #     ans = qa['sent']
        ans_valid = is_ans_valid(ans)

        entry = {'question_id': qa['question_id'],
                 'image_id': qa['img_id'],
                 'image': imd_id2val[qa['img_id']],
                 'question': qa['sent'],
                 'answer': answer,
                 'ans_sent': ans,
                 'ans_valid': ans_valid}
        if not COUNTING_ONLY or is_howmany(qa['sent'], answer, label2ans):
            entries.append(entry)
    entries.sort(key=lambda x: x['question_id'])
    return entries


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '')\
            .replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK
                # for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
        (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))
    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries


def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot,
                                 'visualGenome/question_answers.json')
    image_data_path = os.path.join(dataroot,
                                   'visualGenome/image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' %
                              (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = pickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        # 108,077 images
        _vgv = json.load(open(image_data_path, 'r'))
        vgv = {}
        for _v in _vgv:
            if _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        # used image, used question, total question, out-of-split
        counts = [0, 0, 0, 0]
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if coco_id is not None:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if img_idx is None:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(
                                q['answer'])
                    label = ans2label.get(_answer, None)
                    if label and img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id': q['qa_id'],
                            'image_id': coco_id,
                            'image': img_idx,
                            'question': q['question'],
                            'answer': answer}
                        if not COUNTING_ONLY \
                           or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' %
              (counts[0], len(_vgv), counts[0]/len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' %
              (counts[3], counts[0], counts[3]/counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' %
              (counts[1], counts[2], counts[1]/counts[2]))
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, open(cache_path, 'wb'))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['image_id'] == vgv_id:
            return v['coco_id']
    return None


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, relation_type, dataroot='data',
                 adaptive=False, pos_emb_dim=64, nongt_dim=36):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = adaptive
        prefix = '36'
        if 'test' in name:
            prefix = '_36'

        h5_dataroot = dataroot+"/Bottom-up-features-adaptive"\
            if self.adaptive else dataroot+"/Bottom-up-features-fixed"
        imgid_dataroot = dataroot+"/imgids"

        self.img_id2idx = pickle.load(
            open(os.path.join(imgid_dataroot, '%s%s_imgi_d2idx.pkl' %
                              (name, '' if self.adaptive else prefix)), 'rb'))

        h5_path = os.path.join(h5_dataroot, '%s%s.hdf5' %
                               (name, '' if self.adaptive else prefix))

        print('loading features from h5 file %s' % h5_path)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.normalized_bb = np.array(hf.get('spatial_features'))
            self.bb = np.array(hf.get('image_bb'))
            if "semantic_adj_matrix" in hf.keys() \
               and self.relation_type == "semantic":
                self.semantic_adj_matrix = np.array(
                                        hf.get('semantic_adj_matrix'))
                print("Loaded semantic adj matrix from file...",
                      self.semantic_adj_matrix.shape)
            else:
                self.semantic_adj_matrix = None
                print("Setting semantic adj matrix to None...")
            if "image_adj_matrix" in hf.keys()\
               and self.relation_type == "spatial":
                self.spatial_adj_matrix = np.array(hf.get('image_adj_matrix'))
                print("Loaded spatial adj matrix from file...",
                      self.spatial_adj_matrix.shape)
            else:
                self.spatial_adj_matrix = None
                print("Setting spatial adj matrix to None...")

            self.pos_boxes = None
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.entries = _load_dataset(dataroot, name, self.img_id2idx,
                                     self.label2ans)
        self.tokenize()

        self.tensorize()
        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.normalized_bb.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad to the back of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.normalized_bb = torch.from_numpy(self.normalized_bb)
        self.bb = torch.from_numpy(self.bb)
        if self.semantic_adj_matrix is not None:
            self.semantic_adj_matrix = torch.from_numpy(
                                        self.semantic_adj_matrix).double()
        if self.spatial_adj_matrix is not None:
            self.spatial_adj_matrix = torch.from_numpy(
                                        self.spatial_adj_matrix).double()
        if self.pos_boxes is not None:
            self.pos_boxes = torch.from_numpy(self.pos_boxes)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]

        question = entry['q_token']
        question_id = entry['question_id']
        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = self.spatial_adj_matrix[entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = self.semantic_adj_matrix[entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()
        if not self.adaptive:
            # fixed number of bounding boxes
            features = self.features[entry['image']]
            normalized_bb = self.normalized_bb[entry['image']]
            bb = self.bb[entry["image"]]
        else:
            features = self.features[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            normalized_bb = self.normalized_bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb = self.bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        answer = entry['answer']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, normalized_bb, question, target,\
                question_id, image_id, bb, spatial_adj_matrix,\
                semantic_adj_matrix

        else:
            return features, normalized_bb, question, question_id,\
                question_id, image_id, bb, spatial_adj_matrix,\
                semantic_adj_matrix

    def __len__(self):
        return len(self.entries)

class PVQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, relation_type, dataroot='data/pvqa', adaptive=False, img_v='',
                pos_emb_dim = 64, nongt_dim = 36):
        super(PVQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        ans2label_path = os.path.join(dataroot, 'qas', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'qas', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = False

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s_img_id2idx.pkl' % name), 'rb'))

        #tsv_file = os.path.join(dataroot, 'images/%s%s.csv' % (name, img_v))
        self.image_data = load_tsv(name)
        self.features = np.array([datum['features'] for datum in self.image_data])
        self.spatials = np.array([datum['boxes'] for datum in self.image_data])
        self.img_w = np.array([datum['img_w'] for datum in self.image_data])
        self.img_h = np.array([datum['img_h'] for datum in self.image_data])

        self.bb_norm = np.array([normalize_bbox(datum['img_w'], datum['img_h'], datum['boxes']) for datum in self.image_data])
        
        self.semantic_adj_matrix = None
        print("Setting semantic adj matrix to None...")
        self.spatial_adj_matrix = None
        print("Setting spatial adj matrix to None...")

        self.entries = _load_dataset_pvqa(dataroot, name, self.img_id2idx, self.label2ans, self.ans2label)
        self.tokenize()
        self.tensorize()

        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """

        for entry in self.entries:

            for k, v in {'question': 'q_token', 'ans_sent': 'ans_token'}.items():
                tokens = self.dictionary.tokenize(entry[k], False)
                tokens = tokens[:max_length]
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_length)
                entry[v] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token'],dtype=np.int64))
            entry['q_token'] = question

            answer_token = torch.from_numpy(np.array(entry['ans_token']))
            entry['ans_token'] = answer_token

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'], dtype=np.int64)
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        
        if not self.adaptive:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
            bb_norm = self.bb_norm[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb_norm = self.bb_norm[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        question = entry['q_token']
        question_id = entry['question_id']
        image_id = entry["image_id"]
        answer = entry['answer']

        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = self.spatial_adj_matrix[entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = self.semantic_adj_matrix[entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, bb_norm, question, target,\
                question_id, image_id, spatials, spatial_adj_matrix,\
                semantic_adj_matrix
        else:
            return features, bb_norm, question, question_id, \
                question_id, image_id, spatials, spatial_adj_matrix, \
                semantic_adj_matrix

    def __len__(self):
        return len(self.entries)


class PretrainInputExample(object):

    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, match_question=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.match_question = match_question
        self.label = label


class PretrainDataset(Dataset):
    def __init__(self, dataset, task, pretrain_tasks=[]):
        super(PretrainDataset, self).__init__()
        print('Pratrain Dataset with tasks: ', ', '.join(pretrain_tasks))
        self.task = task
        self.dataset = dataset
        self.pretrain_tasks = pretrain_tasks
        pass

    def __getitem__(self, index):
        datum = self.dataset[index]
        feats, spatials, question, l = datum
        entry = self.dataset.entries[index]
        label = l if entry['answer'] is not None else question
        ans_valid = entry['ans_valid']

        """
        entry = {'question_id': qa['question_id'],
             'image_id': qa['img_id'],
             'image': imd_id2val[qa['img_id']],
             'question': qa['sent'],
             'answer': answer}
        
        datum: features, spatials, question, target 
        """
        # print('entry:')
        # print(entry)
        uid = entry['question_id']
        # question in token form
        feats = feats.clone()
        spatials = spatials.clone()
        # obj_labels = None
        # obj_confs = None
        # attr_labels = None
        # attr_confs = None

        vq_matched = [0.0, 1.0]
        match_question = question.clone()
        if 'vq' in self.pretrain_tasks:
            if np.random.random() < 0.5:
                vq_matched = [1.0, 0.0]
                other_idx = np.random.randint(0, len(self.dataset))
                while self.dataset.entries[other_idx]['image_id'] == entry['image_id']:
                    other_idx = np.random.randint(0, len(self.dataset))
                match_question = self.dataset[other_idx][2].clone()

        if 'qa' in self.pretrain_tasks:
            """qa does not need additional data"""
            pass

        va_matched = [0.0, 1.0]
        answer_rps = entry['ans_token']
        ans_rps_valid = ans_valid
        if 'va' in self.pretrain_tasks:
            if np.random.random() < 0.5:
                va_matched = [1.0, 0.0]
                other_idx = np.random.randint(0, len(self.dataset))
                while self.dataset.entries[other_idx]['image_id'] == entry['image_id']:
                    other_idx = np.random.randint(0, len(self.dataset))
                answer_rps = self.dataset.entries[other_idx]['ans_token']
                ans_rps_valid = self.dataset.entries[other_idx]['ans_valid']

        if 'va2' in self.pretrain_tasks:
            """va2 does not need additional data"""
            pass

        """example = PretrainInputExample(uid, question, (feats, spatials), (obj_labels, obj_confs),
                                       (attr_labels, attr_confs),
                                       is_matched, match_question, label
                                       )
        """
        # print('example')

        example = (uid, question, (feats, spatials),
                   torch.tensor(vq_matched), match_question, torch.tensor(va_matched), answer_rps, label,
                   torch.tensor(ans_valid), torch.tensor(ans_rps_valid))
        # print(example)
        return example

    def __len__(self):
        return len(self.dataset)


question_types = ('where', 'what', 'how', 'how many/how much', 'when', 'why', 'who/whose', 'other', 'yes/no')


def get_q_type(q: str):
    q = q.lower()
    if q.startswith('how many') or q.startswith('how much'):
        return 'how many/how much'
    first_w = q.split()[0]
    if first_w in ('who', 'whose'):
        return 'who/whose'
    for q_type in ('where', 'what', 'how', 'when', 'why'):
        if first_w == q_type:
            return q_type
    if first_w in ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'does', 'do', 'did', 'can', 'could']:
        return 'yes/no'
    if 'whose' in q:
        return 'who/whose'
    if 'how many' in q or 'how much' in q:
        return 'how many/how much'
    for q_type in ('where', 'what', 'how', 'when', 'why'):
        if q_type in q:
            return q_type
    print(q)
    return 'other'


class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, name, features, normalized_bb, bb,
                 spatial_adj_matrix, semantic_adj_matrix, dictionary,
                 relation_type, dataroot='data', adaptive=False,
                 pos_boxes=None, pos_emb_dim=64):
        super(VisualGenomeFeatureDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']
        print('loading Visual Genome data %s' % name)
        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, 'imgids/%s%s_imgid2idx.pkl' %
                                  (name, '' if self.adaptive else '36')),
                     'rb'))
        self.bb = bb
        self.features = features
        self.normalized_bb = normalized_bb
        self.spatial_adj_matrix = spatial_adj_matrix
        self.semantic_adj_matrix = semantic_adj_matrix

        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_visualgenome(dataroot, name, self.img_id2idx,
                                          self.label2ans,
                                          adaptive=self.adaptive)
        self.tokenize()
        self.tensorize()
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.normalized_bb.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = self.spatial_adj_matrix[entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = self.semantic_adj_matrix[entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()
        if self.adaptive:
            features = self.features[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            normalized_bb = self.normalized_bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb = self.bb[self.pos_boxes[
                entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        else:
            features = self.features[entry['image']]
            normalized_bb = self.normalized_bb[entry['image']]
            bb = self.bb[entry['image']]

        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, normalized_bb, question, target, raw_question,\
            image_id, bb, spatial_adj_matrix, semantic_adj_matrix

    def __len__(self):
        return len(self.entries)


def tfidf_from_questions(names, dictionary, dataroot='data',
                         target=['vqa', 'vg']):
    # rows, cols for uncoalesce sparse matrix
    inds = [[], []]
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1])
                inds[1].append(c[0])

    # VQA 2.0
    if 'vqa' in target:
        for name in names:
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
                (name + '2014' if 'test' != name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    # Visual Genome
    if 'vg' in target:
        question_path = os.path.join(dataroot, 'visualGenome',
                                     'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    # TF-IDF
    vals = np.ones((len(inds[1])))
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds),
                                     torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = dataroot+'/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(
                        dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0),
          tfidf.size(1)))

    return tfidf, weights


# VisualGenome Train
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 17464/51487 (0.3392)
#     Used VG questions: 325311/726932 (0.4475)

# VisualGenome Val
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 34023/51487 (0.6608)
#     Used VG questions: 166409/726932 (0.2289)

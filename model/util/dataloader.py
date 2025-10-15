import numpy as np
import os
import torch
import torch.utils.data as data
import json
import threading
import sys

# from transformers import BertTokenizer
from lxrt.tokenization import BertTokenizer

# convert problematic answer
ANS_CONVERT = {
    "a man": "man",
    "the man": "man",
    "a woman": "woman",
    "the woman": "woman",
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    'grey': 'gray',
}

ANS_REMOVED = [

]

detected_ans = ['yes', 'no', 'dark', 'brown', 'blond', 'blue', 'green', 'yellow', 'black', 'gray', 'white', 'red',
                'purple', 'light brown', 'light blue', 'silver', 'pink', 'orange', 'tan', 'dark brown', 'beige', 'gold',
                'cream colored', 'maroon', 'dark blue', 'khaki', 'teal', 'brunette', 'left', 'right', 'large', 'small',
                'huge', 'little', 'giant', 'tiny', 'young', 'old', 'long', 'short', 'tall']


def convert_sents_to_features(sent, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    tokens_a = tokenizer.tokenize(sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # Keep segment id which allows loading BERT-weights.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long), \
           torch.tensor(segment_ids, dtype=torch.long)


class Batch_generator(data.Dataset):
    def __init__(self, img_dir, que_dir, lang_dir, box_dir, max_qlen=18, max_exp_len=18, seq_len=20,
                 mode='train', percentage=100, explainable=True):
        self.mode = mode
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.max_qlen = max_qlen  # maximum len of question
        self.max_exp_len = max_exp_len
        self.seq_len = seq_len
        self.explainable = explainable
        # selecting top answers
        self.ans2idx = json.load(open(os.path.join(lang_dir, 'ans2idx_all.json')))
        self.exp2idx = json.load(open(os.path.join(lang_dir, 'exp2idx.json')))
        if self.mode == 'train_submission':
            self.answer_mask = None
        else:
            self.answer_mask = json.load(open(os.path.join(lang_dir, 'answer_mask_{}.json'.format(mode))))

        self.vanilla_mask_ans = np.zeros(len(self.ans2idx), dtype=np.bool)
        vanilla_mask_ans_id = [self.ans2idx[ans] for ans in detected_ans]
        self.vanilla_mask_ans[vanilla_mask_ans_id] = True

        if self.mode == 'train' and percentage != 100:
            self.question = json.load(open(os.path.join(que_dir, 'train_balanced_questions_clean' + str(percentage) + '.json')))
        elif self.mode == 'train_submission':
            self.question = dict()
            for split in ['train_all_0', 'train_all_1', 'train_all_2', 'train_all_3', 'train_all_4', 'train_all_5',
                          'train_all_6', 'train_all_7', 'train_all_8', 'train_all_9', 'val_all']:
                cur_question = json.load(open(os.path.join(que_dir, split + '_questions_clean.json')))
                cur_question = {qid: {'question': cur_question[qid]['question'], 'imageId': cur_question[qid]['imageId'],
                                'answer': cur_question[qid]['answer']} for qid in cur_question.keys()}
                self.question.update(cur_question)
        else:
            self.question = json.load(open(os.path.join(que_dir, mode + '_questions_clean.json')))

        if 'testdev' not in self.mode and explainable:
            self.explanation = json.load(open(os.path.join(lang_dir, 'converted_explanation_' + mode + '.json')))
        else:
            self.explanation = dict()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.init_data()

    def init_data(self, ):
        self.Q = []
        self.answer = []
        self.Img = []
        self.Qid = []
        self.pure_answer = dict()
        self.converted_exp = []
        self.structure_gate = []
        self.objs = []
        self.attrs = []
        self.ansm = []

        for qid in self.question.keys():
            # convert question
            cur_Q = self.question[qid]['question']  # .replace(',', ' ').replace('.', '')
            # cur_Q = [cur for cur in cur_Q.split(' ') if cur not in ['', ' ']]
            # cur_Q = ' '.join(cur_Q)

            # remove questions that exceed specific length, originally 18
            # if len(cur_Q.split(' ')) > self.max_qlen and self.mode == 'train':
            if len(self.tokenizer.tokenize(cur_Q.strip())) > self.max_qlen and self.mode == 'train':
                continue

            cur_img = self.question[qid]['imageId']

            cur_A = self.question[qid]['answer']
            if cur_A in ANS_CONVERT:
                cur_A = ANS_CONVERT[cur_A]
            self.pure_answer[qid] = cur_A
            if cur_A in self.ans2idx:
                cur_A = self.ans2idx[cur_A]
            else:
                continue

            if 'train' in self.mode and self.explainable:
                if qid in self.explanation:
                    raw_exp = self.explanation[qid].replace('?', '').replace(',', ' ').replace('.', '').split(' ')
                else:
                    raw_exp = ''
                raw_exp = [self.exp2idx[cur] for cur in raw_exp if cur not in ['', ' ']]
                structure_gate = [0 if cur in range(1, 37) else 1 for cur in raw_exp]
                if len(raw_exp) > self.max_exp_len:
                    continue
                self.converted_exp.append(raw_exp)
                self.structure_gate.append(structure_gate)

            if 'train' not in self.mode and qid not in self.explanation and self.explainable:
                self.explanation[qid] = ''

            self.Q.append(cur_Q)
            self.answer.append(cur_A)
            self.Img.append(cur_img)
            self.Qid.append(qid)
            if self.answer_mask is not None:
                self.ansm.append(self.answer_mask[qid])

    def eval_qa_score(self, pred, qid_list):
        acc = []
        gt = []
        for i, qid in enumerate(qid_list):
            cur_gt = self.pure_answer[qid]
            if cur_gt == pred[i]:
                acc.append(1)
            else:
                acc.append(0)
            gt.append(cur_gt)
        return acc, gt

    def __getitem__(self, index):
        question = self.Q[index]
        img_id = self.Img[index]
        qid = self.Qid[index]

        if self.answer_mask is not None:
            if len(self.ansm[index]) > 0:
                mask_ans = torch.ones(len(self.ans2idx), dtype=torch.bool)
                mask_ans[self.ansm[index]] = False
            else:
                mask_ans = torch.tensor(self.vanilla_mask_ans)
        else:
            mask_ans = torch.zeros((1))

        # merging question and explanation mask for inputs
        # encode_question = self.tokenizer(question, return_tensors='pt')
        # text_input = encode_question['input_ids'][0]
        # token_type = encode_question['token_type_ids'][0]
        # attention_mask = encode_question['attention_mask'][0]
        text_input, attention_mask, token_type = convert_sents_to_features(question, self.seq_len, self.tokenizer)

        text_input = text_input[:self.seq_len]
        token_type = token_type[:self.seq_len]
        attention_mask = attention_mask[:self.seq_len]

        # padding
        if len(text_input) < self.seq_len:
            pad_len = self.seq_len - len(text_input)
            pads = torch.zeros(pad_len)
            text_input = torch.cat((text_input, pads), dim=0)
            token_type = torch.cat((token_type, pads), dim=0)
            attention_mask = torch.cat((attention_mask, pads), dim=0)

        # load image features
        img = np.load(os.path.join(self.img_dir, str(img_id) + '.npy'))
        box = np.load(os.path.join(self.box_dir, str(img_id) + '.npy'))
        # img = np.concatenate((img, np.zeros((36, 6), dtype=np.float32)), axis=1)

        if 'train' in self.mode:
            answer = self.answer[index]
            converted_ans = torch.zeros((len(self.ans2idx),))
            converted_ans[answer] = 1

            if self.explainable:
                exp = self.converted_exp[index]
                converted_exp = torch.zeros(self.max_exp_len, len(self.exp2idx) + 1)
                valid_mask = torch.zeros(self.max_exp_len)
                structure_gate = self.structure_gate[index]
                structure_gate = structure_gate[:self.max_exp_len]
                converted_gate = np.ones([self.max_exp_len, ]).astype('float32')

                for i in range(len(exp)):
                    converted_exp[i, exp[i]] = 1
                    converted_gate[i] = structure_gate[i]
                valid_mask[:len(exp) + 1] = 1

                if len(exp) < self.max_exp_len:
                    converted_exp[len(exp), 0] = 1
            else:
                converted_exp = torch.zeros((1))
                valid_mask = torch.zeros((1))
                converted_gate = torch.zeros((1))

            return img, box, text_input.long(), token_type.long(), attention_mask.long(), converted_ans, converted_exp, \
                   valid_mask, converted_gate, mask_ans
        else:
            return img, box, text_input.long(), token_type.long(), attention_mask.long(), qid, mask_ans

    def __len__(self, ):
        return len(self.Img)



class Batch_generator_submission(data.Dataset):
    def __init__(self, img_dir, que_dir, lang_dir, box_dir, seq_len=20, mode='submission_all'):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.box_dir = box_dir
        # selecting top answers
        self.ans2idx = json.load(open(os.path.join(lang_dir, 'ans2idx_all.json')))
        self.exp2idx = json.load(open(os.path.join(lang_dir, 'exp2idx.json')))

        # self.question = json.load(open(os.path.join(que_dir, 'submission_all_questions_clean.json')))
        self.question = json.load(open(os.path.join(que_dir, '{}_questions_clean.json'.format(mode))))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.answer_mask = json.load(open(os.path.join(lang_dir, 'answer_mask_{}.json'.format(mode))))
        self.vanilla_mask_ans = np.zeros(len(self.ans2idx), dtype=np.bool)
        vanilla_mask_ans_id = [self.ans2idx[ans] for ans in detected_ans]
        self.vanilla_mask_ans[vanilla_mask_ans_id] = True

        self.init_data()

    def init_data(self, ):
        self.Q = []
        self.Img = []
        self.Qid = []
        self.ansm = []

        for qid in self.question.keys():
            # convert question
            cur_Q = self.question[qid]['question']  # .replace(',', ' ').replace('.', '')
            # cur_Q = [cur for cur in cur_Q.split(' ') if cur not in ['', ' ']]
            # cur_Q = ' '.join(cur_Q)

            cur_img = self.question[qid]['imageId']

            self.Q.append(cur_Q)
            self.Img.append(cur_img)
            self.Qid.append(qid)
            self.ansm.append(self.answer_mask[qid])

    def __getitem__(self, index):
        question = self.Q[index]
        img_id = self.Img[index]
        qid = self.Qid[index]

        if len(self.ansm[index]) > 0:
            mask_ans = torch.ones(len(self.ans2idx), dtype=torch.bool)
            mask_ans[self.ansm[index]] = False
        else:
            mask_ans = torch.tensor(self.vanilla_mask_ans)

        text_input, attention_mask, token_type = convert_sents_to_features(question, self.seq_len, self.tokenizer)

        text_input = text_input[:self.seq_len]
        token_type = token_type[:self.seq_len]
        attention_mask = attention_mask[:self.seq_len]

        # padding
        pad_len = 0
        if len(text_input) < self.seq_len:
            pad_len = self.seq_len - len(text_input)
            pads = torch.zeros(pad_len)
            text_input = torch.cat((text_input, pads), dim=0)
            token_type = torch.cat((token_type, pads), dim=0)
            attention_mask = torch.cat((attention_mask, pads), dim=0)

        # load image features
        img = np.load(os.path.join(self.img_dir, str(img_id) + '.npy'))
        box = np.load(os.path.join('../preprocessing/data/extracted_features/box', str(img_id) + '.npy'))

        return img, box, text_input.long(), token_type.long(), attention_mask.long(), qid, mask_ans

    def __len__(self, ):
        return len(self.Img)


# convert words within string to index
def convert_idx(sentence, word2idx):
    idx = []
    for word in sentence:
        if word in word2idx:
            idx.append(word2idx[word])
        else:
            idx.append(word2idx['UNK'])

    return idx

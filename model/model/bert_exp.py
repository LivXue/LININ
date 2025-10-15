import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisualBertModel

from modules import GaussianEncoder
from exp_generator import ExpGenerator, LSTMGenerator, ProgramEncoder
from lxrt.entry import LXRTEncoder
from loss import exp_generative_loss, structure_bce, kl_div_gaussian


class GRU(nn.Module):
    """
    Gated Recurrent Unit without long-term memory
    """

    def __init__(self, input_size, embed_size=512):
        super(GRU, self).__init__()
        self.update_x = nn.Linear(input_size, embed_size, bias=True)
        self.update_h = nn.Linear(embed_size, embed_size, bias=True)
        self.reset_x = nn.Linear(input_size, embed_size, bias=True)
        self.reset_h = nn.Linear(embed_size, embed_size, bias=True)
        self.memory_x = nn.Linear(input_size, embed_size, bias=True)
        self.memory_h = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, x, state):
        z = torch.sigmoid(self.update_x(x) + self.update_h(state))
        r = torch.sigmoid(self.reset_x(x) + self.reset_h(state))
        mem = torch.tanh(self.memory_x(x) + self.memory_h(torch.mul(r, state)))
        state = torch.mul(1 - z, state) + torch.mul(z, mem)
        return state


class VisualBert_REX(nn.Module):
    """
    Baseline method
    """
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=12, use_structure=True, lang_dir=None):
        super(VisualBert_REX, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.use_structure = use_structure
        base_model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        self.embedding = base_model.embeddings
        self.bert_encoder = base_model.encoder
        self.sent_cls = nn.Linear(768, self.nb_vocab + 1)
        self.ans_cls = nn.Linear(768, self.nb_answer)

        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab + 1, embedding_dim=self.hidden_size, padding_idx=0)

        # attentive RNN
        self.att_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.att_v = nn.Linear(self.img_size, self.hidden_size)
        self.att_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.att = nn.Linear(self.hidden_size, 1)
        self.att_rnn = GRU(3 * self.hidden_size, self.hidden_size)

        # language RNN
        self.q_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_fc = nn.Linear(self.img_size, self.hidden_size)

        self.language_rnn = GRU(2 * self.hidden_size, self.hidden_size)
        self.language_fc = nn.Linear(self.hidden_size, nb_vocab + 1)
        # self.att_drop = nn.Dropout(0.2)
        # self.fc_drop = nn.Dropout(0.2)

        if self.use_structure:
            self.structure_gate = nn.Linear(self.hidden_size, 1)
            self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                                  requires_grad=False)

        for module in [self.embedding, self.bert_encoder]:
            for para in module.parameters():
                para.requires_grad = True  # fixed pretrained or not

    def create_att_mask(self, batch, ori_mask, device):
        visual_mask = torch.ones(batch, self.num_roi).to(device)
        mask = torch.cat((ori_mask, visual_mask,), dim=1)
        return mask

    def init_hidden_state(self, batch, device):
        init_word = torch.zeros(batch, self.hidden_size).to(device)
        init_att_h = torch.zeros(batch, self.hidden_size).to(device)
        init_language_h = torch.zeros(batch, self.hidden_size).to(device)
        return init_word, init_att_h, init_language_h

    def forward(self, img, box, text_input, token_type, attention_mask, exp=None):
        embedding = self.embedding(input_ids=text_input, token_type_ids=token_type, visual_embeds=img)
        visual_mask = torch.ones(len(embedding), self.num_roi).cuda()
        concat_mask = torch.cat((attention_mask, visual_mask,), dim=1)
        # concat_mask = self.create_att_mask(len(embedding),attention_mask, embedding.device)

        # manually create attention mask for bert encoder (copy from PreTrainedModel's function)
        extended_mask = concat_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        bert_feat = self.bert_encoder(embedding, extended_mask)[0]
        visual_feat = bert_feat[:, -int(self.num_roi):, :].contiguous()
        cls_feat = bert_feat[:, 0]

        # pre-computed features for attention computation
        v_att = torch.tanh(self.att_v(img))
        q_att = torch.tanh(self.att_q(cls_feat))
        q_att = q_att.view(q_att.size(0), 1, -1)
        fuse_feat = torch.mul(v_att, q_att.expand_as(v_att))

        # pre-compute features for language prediction
        q_enc = torch.tanh(self.q_fc(cls_feat))

        # initialize hidden state
        prev_word = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        # h_1 = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        h_2 = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        # prev_word, h_1, h_2 = self.init_hidden_state(len(fuse_feat), fuse_feat.device)

        # loop for explanation generation
        pred_exp = []
        pred_gate = []
        pred_att = []
        x_1 = torch.cat((fuse_feat.mean(1), h_2, prev_word), dim=-1)
        for i in range(self.num_step):
            # attentive RNN
            h_1 = self.att_rnn(x_1, h_2)
            att_h = torch.tanh(self.att_h(h_1).unsqueeze(1).expand_as(fuse_feat) + fuse_feat)
            att = F.softmax(self.att(att_h), dim=1)  # with dropout
            # pred_att.append(att.squeeze(1))

            # use separate layers to encode the attended features
            att_x = torch.bmm(att.transpose(1, 2).contiguous(), img).squeeze()
            v_enc = torch.tanh(self.v_fc(att_x))
            fuse_enc = torch.mul(v_enc, q_enc)

            x_2 = torch.cat((fuse_enc, h_1), dim=-1)

            # language RNN
            h_2 = self.language_rnn(x_2, h_2)
            pred_word = F.softmax(self.language_fc(h_2), dim=-1)  # without dropout

            if self.use_structure:
                structure_gate = torch.sigmoid(self.structure_gate(h_2))
                similarity = torch.bmm(h_2.unsqueeze(1), visual_feat.transpose(1, 2)).squeeze(1)
                similarity = F.softmax(similarity, dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.nb_vocab + 1,
                                                                               self.num_roi)
                sim_pred = torch.bmm(structure_mapping, similarity.unsqueeze(-1)).squeeze(-1)
                pred_word = structure_gate * pred_word + (1 - structure_gate) * sim_pred
                pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # schedule sampling
            prev_word = torch.max(pred_word, dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1), h_2, prev_word), dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp], dim=1)
        output_ans = F.softmax(self.ans_cls(cls_feat), dim=-1)
        # output_att = torch.cat([_.unsqueeze(1) for _ in pred_att],dim=1)
        output_gate = torch.cat([_ for _ in pred_gate], dim=1)

        if self.training:
            return output_ans, output_sent, output_gate
        else:
            return output_ans, output_sent


class LXMERT_REX(nn.Module):
    """
    Baseline method based on LXMERT
    """
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=12, use_structure=True, lang_dir=None):
        super(LXMERT_REX, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.use_structure = use_structure
        base_model = LXRTEncoder(max_seq_length=18)
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.sent_cls = nn.Linear(768, self.nb_vocab + 1)
        self.ans_cls = nn.Linear(768, self.nb_answer)

        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab + 1, embedding_dim=self.hidden_size, padding_idx=0)

        # attentive RNN
        self.att_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.att_v = nn.Linear(self.img_size, self.hidden_size)
        self.att_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.att = nn.Linear(self.hidden_size, 1)
        self.att_rnn = GRU(3 * self.hidden_size, self.hidden_size)

        # language RNN
        self.q_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_fc = nn.Linear(self.img_size, self.hidden_size)

        self.language_rnn = GRU(2 * self.hidden_size, self.hidden_size)
        self.language_fc = nn.Linear(self.hidden_size, nb_vocab + 1)
        # self.att_drop = nn.Dropout(0.2)
        # self.fc_drop = nn.Dropout(0.2)

        self.v0 = nn.Parameter(torch.zeros((1, 36, 2048)))
        self.b0 = nn.Parameter(torch.zeros((1, 36, 4)))

        if self.use_structure:
            self.structure_gate = nn.Linear(self.hidden_size, 1)
            self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                                  requires_grad=False)

        #for module in [self.embedding, self.bert_encoder]:
        #    for para in module.parameters():
        #        para.requires_grad = True  # fixed pretrained or not

    def create_att_mask(self, batch, ori_mask, device):
        visual_mask = torch.ones(batch, self.num_roi).to(device)
        mask = torch.cat((ori_mask, visual_mask,), dim=1)
        return mask

    def init_hidden_state(self, batch, device):
        init_word = torch.zeros(batch, self.hidden_size).to(device)
        init_att_h = torch.zeros(batch, self.hidden_size).to(device)
        init_language_h = torch.zeros(batch, self.hidden_size).to(device)
        return init_word, init_att_h, init_language_h

    def forward(self, img, box, text_input, token_type, attention_mask, exp=None):
        visual_mask = torch.ones(len(img), self.num_roi).cuda()
        v00 = self.v0.expand(img.size()[0], -1, -1)
        b00 = self.b0.expand(img.size()[0], -1, -1)

        feat_seq, cls_feat = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                               attention_mask=attention_mask, visual_feats=(img, box),
                                               visual_attention_mask=visual_mask)
        _, cls_feat0 = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                         attention_mask=attention_mask, visual_feats=(v00, b00),
                                         visual_attention_mask=visual_mask)
        que_feat = feat_seq[0]
        visual_feat = feat_seq[1]

        # pre-computed features for attention computation
        v_att = torch.tanh(self.att_v(img))
        q_att = torch.tanh(self.att_q(cls_feat))
        q_att = q_att.view(q_att.size(0), 1, -1)
        fuse_feat = torch.mul(v_att, q_att.expand_as(v_att))

        # pre-compute features for language prediction
        q_enc = torch.tanh(self.q_fc(cls_feat))

        # initialize hidden state
        prev_word = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        # h_1 = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        h_2 = torch.zeros(len(fuse_feat), self.hidden_size).cuda()
        # prev_word, h_1, h_2 = self.init_hidden_state(len(fuse_feat), fuse_feat.device)

        # loop for explanation generation
        pred_exp = []
        pred_gate = []
        x_1 = torch.cat((fuse_feat.mean(1), h_2, prev_word), dim=-1)
        for i in range(self.num_step):
            # attentive RNN
            h_1 = self.att_rnn(x_1, h_2)
            att_h = torch.tanh(self.att_h(h_1).unsqueeze(1).expand_as(fuse_feat) + fuse_feat)
            att = F.softmax(self.att(att_h), dim=1)  # with dropout
            # pred_att.append(att.squeeze(1))

            # use separate layers to encode the attended features
            att_x = torch.bmm(att.transpose(1, 2).contiguous(), img).squeeze()
            v_enc = torch.tanh(self.v_fc(att_x))
            fuse_enc = torch.mul(v_enc, q_enc)

            x_2 = torch.cat((fuse_enc, h_1), dim=-1)

            # language RNN
            h_2 = self.language_rnn(x_2, h_2)
            pred_word = F.softmax(self.language_fc(h_2), dim=-1)  # without dropout

            if self.use_structure:
                structure_gate = torch.sigmoid(self.structure_gate(h_2))
                similarity = torch.bmm(h_2.unsqueeze(1), visual_feat.transpose(1, 2)).squeeze(1)
                similarity = F.softmax(similarity, dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.nb_vocab + 1,
                                                                               self.num_roi)
                sim_pred = torch.bmm(structure_mapping, similarity.unsqueeze(-1)).squeeze(-1)
                pred_word = structure_gate * pred_word + (1 - structure_gate) * sim_pred
                pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # schedule sampling
            prev_word = torch.max(pred_word, dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1), h_2, prev_word), dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp], dim=1)
        output_ans = F.softmax(self.ans_cls(cls_feat), dim=-1)
        output_gate = torch.cat([_ for _ in pred_gate], dim=1)

        if self.training:
            output_ans = F.softmax(self.ans_cls(cls_feat + cls_feat0), dim=-1)
            return output_ans, output_sent, output_gate
        else:
            output_ans = F.softmax(self.ans_cls(cls_feat + 0.25 * cls_feat0), dim=-1)
            return output_ans, output_sent


class LXMERTRegion(nn.Module):
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, nb_pro=2000, num_step=18, lang_dir=None, args=None, explainable=True):
        super(LXMERTRegion, self).__init__()
        self.nb_vocab = nb_vocab
        self.nb_pro = nb_pro
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.num_head = 4
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)     # "max_seq_length" not used
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                     nn.GELU(),
                                     nn.LayerNorm(768 * 2),
                                     nn.Linear(768 * 2, self.nb_answer))

        # language generator
        if explainable:
            self.exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)

    def forward(self, img, box, text_input, token_type, attention_mask, mask_ans=None, exp=None):
        visual_mask = torch.ones(len(img), self.num_roi).to(img.device)
        concat_mask = torch.cat((attention_mask, visual_mask,), dim=1)

        feat_seq, pooled_output, hie_visn_feats = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_mask)
        que_feat = feat_seq[0]
        visual_feat = feat_seq[1]
        cls_feat = pooled_output

        if mask_ans is None:
            mask_ans = 1

        if self.explainable:
            pred_pro, pred_output, structure_gates = self.exp_generator(exp, que_feat, visual_feat, concat_mask)
            output_ans_final = self.ans_cls(cls_feat) - (1 - mask_ans) * 1e6
        else:
            pred_pro, structure_gates = None, None
            output_ans_final = self.ans_cls(cls_feat)
        
        return output_ans_final, pred_pro, structure_gates

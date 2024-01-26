import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, use_res=True):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.use_res = use_res

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            r_att = att.masked_fill(attention_mask.unsqueeze(1), -np.inf)
            # mask before softmax, so exp(-inf) will be 0
        else:
            r_att = att
        att = torch.softmax(r_att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = self.dropout(out)
        if self.use_res:
            out = self.layer_norm(queries + out)
        else:
            out = self.layer_norm(out)
        return out, r_att


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, use_position_emb=False, max_seq_len=12):
        super(TransformerLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)

        self.use_position_emb = use_position_emb
        if self.use_position_emb:
            tmp = torch.Tensor(max_seq_len, d_model)
            nn.init.xavier_normal_(tmp)
            self.position_emb = nn.Parameter(tmp.unsqueeze(0))

    def forward(self, input, mask=None, only_last=False):
        if self.use_position_emb:
            input = input + self.position_emb[:, -input.size()[1]:]

        if only_last:
            feature, _ = self.self_att(input[:, -1].unsqueeze(1), input, input, mask)
            feature = feature.squeeze(1)
        else:
            feature, _ = self.self_att(input, input, input, mask)
        feature = self.feed_forward(feature)
        feature = self.act(feature)
        if only_last:
            feature = input[:, -1] + feature
        else:
            feature = input + feature
        feature = self.layer_norm(feature)

        return feature


class ExpGenerator(nn.Module):
    def __init__(self, num_roi, nb_vocab, lang_dir, max_len=12):
        super(ExpGenerator, self).__init__()
        self.num_roi = num_roi
        self.max_len = max_len
        self.nb_vocab = nb_vocab
        self.hidden_size = 768
        self.num_head = 4
        self.drop_prob_lm = 0.1

        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab + 2, embedding_dim=self.hidden_size, padding_idx=0)

        self.structure_gate = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.GELU(),
                                            nn.LayerNorm(self.hidden_size),
                                            nn.Linear(self.hidden_size, 1))
        self.structure_mapping = nn.Parameter(
            torch.load(os.path.join(lang_dir, 'structure_mapping.pth')).T.contiguous(), requires_grad=False)

        self.transformer_layer1 = TransformerLayer(self.hidden_size, self.hidden_size // 2, self.hidden_size // 2,
                                                   self.num_head, dropout=self.drop_prob_lm)
        self.att1 = MultiHeadAttention(self.hidden_size, self.hidden_size // 2, self.hidden_size // 2, self.num_head)
        self.transformer_layer2 = TransformerLayer(self.hidden_size, self.hidden_size // 2, self.hidden_size // 2,
                                                   self.num_head, dropout=self.drop_prob_lm)
        self.att2 = MultiHeadAttention(self.hidden_size, self.hidden_size // 2, self.hidden_size // 2, self.num_head)
        self.output_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 2),
                                           nn.GELU(),
                                           nn.LayerNorm(self.hidden_size * 2),
                                           nn.Linear(self.hidden_size * 2, nb_vocab + 1))
        self.que_emb = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.img_emb = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, pre_words, question_features, img_features, concat_mask, R_label=None):
        concat_mask = concat_mask.bool()
        if R_label is not None:
            gate_label = (R_label.sum(1, keepdims=True) == 0).unsqueeze(1)

        question_features = question_features + self.que_emb
        img_features = img_features + self.img_emb
        mm_features = torch.cat((question_features, img_features), dim=1)

        if self.training:
            pre_words = torch.max(pre_words, dim=-1)[1]
            cls = torch.ones_like(concat_mask[:, 0], dtype=torch.long).unsqueeze(1) * (self.nb_vocab + 1)
            pre_words = torch.cat((cls, pre_words), dim=1)[:, :self.max_len]
            pre_word_features = self.exp_embed(pre_words)
            seq_len = pre_words.shape[1]
            exp_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).unsqueeze(0).to(
                question_features.device)
            mm_mask = ~concat_mask.unsqueeze(1).bool()
            word_features = self.transformer_layer1(pre_word_features, exp_mask)
            word_features, _ = self.att1(word_features, mm_features, mm_features, mm_mask)
            word_features = self.transformer_layer2(word_features, exp_mask)
            word_features, _ = self.att2(word_features, mm_features, mm_features, mm_mask)

            similarity = torch.bmm(word_features, img_features.transpose(1, 2))
            if R_label is not None:
                structure_gates = torch.max(torch.sigmoid(self.structure_gate(word_features)), gate_label.float())
                similarity = similarity.masked_fill(~(R_label.unsqueeze(1) + gate_label), -np.inf)
            else:
                structure_gates = torch.sigmoid(self.structure_gate(word_features))
            similarity = F.softmax(similarity, dim=-1)
            structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.num_roi,
                                                                           self.nb_vocab + 1)
            r_mask = torch.bmm(torch.ones_like(similarity), structure_mapping).bool()
            sim_pred = torch.bmm(similarity, structure_mapping)
            pred_pro = torch.softmax(self.output_linear(word_features).masked_fill(r_mask, -np.inf), -1)
            pred_pro = structure_gates * pred_pro + (1 - structure_gates) * sim_pred
            pred_output = torch.max(pred_pro, dim=-1)[1]
        else:
            pred_pro = []
            structure_gates = []
            pre_words = torch.ones_like(concat_mask[:, 0], dtype=torch.long).unsqueeze(1) * (self.nb_vocab + 1)
            for step in range(self.max_len):
                pre_word_features = self.exp_embed(pre_words)
                seq_len = pre_words.shape[1]
                exp_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).unsqueeze(0).to(
                    question_features.device)
                mm_mask = ~concat_mask.unsqueeze(1).bool()
                word_features = self.transformer_layer1(pre_word_features, exp_mask)
                word_features, _ = self.att1(word_features, mm_features, mm_features, mm_mask)
                word_features = self.transformer_layer2(word_features, only_last=True)
                word_features, _ = self.att2(word_features.unsqueeze(1), mm_features, mm_features, mm_mask)

                cur_word_feature = word_features
                similarity = torch.bmm(cur_word_feature, img_features.transpose(1, 2))
                if R_label is not None:
                    structure_gate = torch.max(torch.sigmoid(self.structure_gate(cur_word_feature)), gate_label.float())
                    similarity = similarity.masked_fill(~(R_label.unsqueeze(1) + gate_label), -np.inf)
                else:
                    structure_gate = torch.sigmoid(self.structure_gate(cur_word_feature))
                similarity = F.softmax(similarity, dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.num_roi,
                                                                               self.nb_vocab + 1)
                r_mask = torch.bmm(torch.ones_like(similarity), structure_mapping).bool()
                sim_pred = torch.bmm(similarity, structure_mapping)
                cur_pred_pro = torch.softmax(self.output_linear(cur_word_feature).masked_fill(r_mask, -np.inf), -1)
                cur_pred_pro = structure_gate * cur_pred_pro + (1 - structure_gate) * sim_pred
                structure_gates.append(structure_gate)
                pred_pro.append(cur_pred_pro)
                pred_word = torch.max(cur_pred_pro.squeeze(1), dim=-1)[1]
                pre_words = torch.cat((pre_words, pred_word.unsqueeze(1)), dim=1)

            pred_pro = torch.cat([_ for _ in pred_pro], dim=1)
            structure_gates = torch.cat([_ for _ in structure_gates], dim=1)
            pred_output = pre_words[:, 1:]

        return pred_pro, pred_output, structure_gates.squeeze(-1)


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


class LSTMGenerator(nn.Module):
    def __init__(self, num_roi, nb_vocab, lang_dir, max_len=12):
        super(LSTMGenerator, self).__init__()
        self.num_roi = num_roi
        self.nb_vocab = nb_vocab
        self.img_size = 2048
        self.hidden_size = 768
        self.num_step = max_len

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
        self.structure_gate = nn.Linear(self.hidden_size, 1)
        self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir, 'structure_mapping.pth')),
                                              requires_grad=False)

    def forward(self, img, cls_feat, visual_feat):
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

            structure_gate = torch.sigmoid(self.structure_gate(h_2))
            similarity = torch.bmm(h_2.unsqueeze(1), visual_feat.transpose(1, 2)).squeeze(1)
            similarity = F.softmax(similarity, dim=-1)
            structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity), self.nb_vocab + 1,
                                                                           self.num_roi)
            sim_pred = torch.bmm(structure_mapping, similarity.unsqueeze(-1)).squeeze(-1)
            pred_word = structure_gate * pred_word + (1 - structure_gate) * sim_pred
            pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # sampling
            prev_word = torch.max(pred_word, dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1), h_2, prev_word), dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp], dim=1)
        output_gate = torch.cat([_ for _ in pred_gate], dim=1)

        return output_sent, None, output_gate

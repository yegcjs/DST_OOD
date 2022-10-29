from transformers import BertModel
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from collections import namedtuple


class TrippyModel(nn.Module):
    def __init__(self, bert_path, config, save_memory=False):
        super(TrippyModel, self).__init__()
        self.config = config
        self.num_slots, self.num_labels = len(config.idx2slot), len(config.idx2label)
        self.bert = BertModel.from_pretrained(bert_path)
        if save_memory:
            self.bert.gradient_checkpointing_enable()
        feature_size = self.bert.config.hidden_size + 2 * self.num_slots
        self.gates = nn.ModuleList([nn.Linear(feature_size, self.num_labels) for _ in range(self.num_slots)])
        self.span = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, 2) for _ in range(self.num_slots)])
        self.refer = nn.ModuleList([nn.Linear(feature_size, self.num_slots) for _ in range(self.num_slots)])

    def forward(self, input_ids, attention_mask, token_type_ids, states, informs, **kwargs):  # kwargs as labels
        trippy_model_ret_t = namedtuple("trippy_model_ret",
                                        ["gate_logits", "span_logits", "ref_logits", "gate_loss", "span_loss",
                                         "refer_loss"]
                                        )
        batch_size, seq_len = input_ids.shape
        bert_features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        features = torch.cat([bert_features.pooler_output, states, informs], dim=-1)
        gate_logits = torch.stack([mlp(features) for mlp in self.gates], dim=1)  # bsz, num_slots, num_labels
        span_logits = torch.stack([mlp(bert_features.last_hidden_state) for mlp in self.span], dim=1)
        span_start_logits, span_end_logits = span_logits[:, :, :, 0], span_logits[:, :, :, 1]  # bsz, num_slots, seq_len
        ref_logits = torch.stack([mlp(features) for mlp in self.refer],
                                 dim=1)  # bsz, num_slots, num_slots
        gate_loss, span_loss, ref_loss = None, None, None
        if "gates" in kwargs:
            gate_loss = cross_entropy(gate_logits.view(-1, self.num_labels), kwargs['gates'].view(-1))
        if "span_start" in kwargs and "span_end" in kwargs:
            span_loss = cross_entropy(span_start_logits.view(-1, seq_len), kwargs['span_start'].view(-1), ignore_index=-1) \
                        + cross_entropy(span_end_logits.view(-1, seq_len), kwargs['span_end'].view(-1), ignore_index=-1)
        if "ref_slot" in kwargs:
            ref_loss = cross_entropy(ref_logits.view(-1, self.num_slots), kwargs['ref_slot'].view(-1), ignore_index=-1)

        return trippy_model_ret_t(gate_logits, span_logits, ref_logits, gate_loss, span_loss, ref_loss)

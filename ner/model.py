from ner.attention import BigBirdAttention

import torch
import torch.nn as nn

from transformers import BertModel


class BERT(nn.Module):
    def __init__(self, model_name: str, classes: int):
        super(BERT, self).__init__()

        self.model_name = model_name

        if self.model_name == 'bert-base-cased':
            self.embedding_dim = 768
        elif self.model_name == 'bert-large-cased':
            self.embedding_dim = 1024
        else:
            raise ValueError('Model name is not valid.')

        self.classes = classes

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.linear = nn.Linear(self.embedding_dim, self.classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        predictions = self.dropout(last_hidden_state)
        predictions = self.linear(predictions)

        return predictions


class BertSelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LongAttentionBERT(nn.Module):
    def __init__(self, model_name: str, classes: int, attention_config=None):
        super(LongAttentionBERT, self).__init__()
        self.model_name = model_name
        self.big_bird_config = attention_config
        self.output = BertSelfOutput()

        if self.model_name == 'bert-base-cased':
            self.embedding_dim = 768
        elif self.model_name == 'bert-large-cased':
            self.embedding_dim = 1024
        else:
            raise ValueError('Model name is not valid.')

        self.classes = classes

        self.sparse_attention = BigBirdAttention(config=self.big_bird_config)

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.linear = nn.Linear(self.embedding_dim, self.classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)[2]

        # выравниваем каждый слой в hidden_states в 1 ряд [num_layers, batch_size * seqlen, embedding_dim]
        hidden_state_one_row = torch.stack(
            [torch.cat(torch.tensor_split(hidden_state, sections=hidden_state.shape[0]), dim=1).squeeze() for
             hidden_state in hidden_states])
        attention_mask_one_row = torch.cat(torch.tensor_split(attention_mask, sections=attention_mask.shape[0]), dim=1).squeeze()
        # last_hidden_state.shape[0] вместо BatchSize,
        # потому что последний батч может быть остатком от деления и != BatchSize
        attention_output = self.sparse_attention(hidden_state_one_row, attention_mask=attention_mask_one_row)[0]  # , attention_mask=attention_mask_one_row)[0]
        # на выходе получаем [num_layers, batch_size * seqlen, embedding_dim], берем последний слой
        # размера [batch_size * seqlen, embedding_dim] и превращаем его в [1, batch_size * seqlen, embedding_dim]
        output = self.output(attention_output)[-1].unsqueeze(0)
        # превращаем этот hidden_state в размерность батча [batch_size, seq_len, embedding_dim]
        batch_output = torch.cat(torch.tensor_split(output, sections=hidden_states[-1].shape[0], dim=1), dim=0)
        predictions = self.linear(batch_output)

        return predictions
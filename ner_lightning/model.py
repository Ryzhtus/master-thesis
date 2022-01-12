import torch.nn as nn

from transformers import BertPreTrainedModel, BertConfig, BertModel


class BERTHeadClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate=0.2, act_func="gelu", intermediate_hidden_size=1024):
        super(BERTHeadClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func
        self.gelu = nn.GELU()

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = self.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

class BERT(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(BERT, self).__init__(config)

        self.bert = BertModel(config)
        self.classes = 9
        self.hidden_size = config.hidden_size
        self.intermediate_hidden_size = 1024
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = BERTHeadClassifier(self.hidden_size, self.classes,
                                             config.hidden_dropout_prob,
                                             act_func='gelu',
                                             intermediate_hidden_size=self.intermediate_hidden_size)
        
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        predictions = self.dropout(last_hidden_state)
        logits = self.classifier(predictions)

        return logits
import torch 
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class Classifier(nn.Module):
    def __init__(self, features: bool, hidden_size: int, classes: int) -> None:
        super(Classifier, self).__init__()

        self.features = features
        self.hidden_size = hidden_size
        self.classes = classes
        self.bilstm_hidden_size = 256
        
        if self.features:
            self.bilstm = nn.LSTM(input_size=self.hidden_size + 1, hidden_size=256, bidirectional=True, batch_first=True)
        else:
            self.bilstm = nn.LSTM(input_size=self.hidden_size + 1, hidden_size=256, bidirectional=True, batch_first=True)
        
        self.classifier = nn.Linear(256 * 2, self.classes)

    def forward(self, x):
        x = self.bilstm(x)[0]
        logits = self.classifier(x)

        return logits


class Baseline(BertPreTrainedModel):
    def __init__(self, config):
        super(Baseline, self).__init__(config)

        self.bert = BertModel(config)
        self.classes = 73
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classification_head = Classifier(features=False, hidden_size=config.hidden_size, classes=self.classes)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classification_head(last_hidden_state)

        return logits 


class BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT, self).__init__(config)

        self.bert = BertModel(config)
        self.classes = 73
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classification_head = Classifier(features=True, hidden_size=config.hidden_size, classes=self.classes)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, features=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        last_hidden_state = self.dropout(last_hidden_state)
        
        word_representations = torch.cat([last_hidden_state, features], dim=-1)
        logits = self.classification_head(word_representations)
        
        return logits 

class BERTLinear(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTLinear, self).__init__(config)

        self.bert = BertModel(config)
        self.classes = 17
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_transform = nn.Linear(3, 3)
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(self.hidden_size + 3, self.classes)

    def forward(self, input_ids, attention_mask=None, features=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        last_hidden_state = self.dropout(last_hidden_state)

        features_output = self.linear_transform(features)
        features_output = self.gelu(features_output)

        concat_output = torch.cat([last_hidden_state, features_output], dim=-1)

        logits = self.classifier(concat_output)
        
        return logits 


class BERTBase(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTBase, self).__init__(config)

        self.bert = BertModel(config)
        self.classes = 17
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.classes)

    def forward(self, input_ids, attention_mask=None, features=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        last_hidden_state = self.dropout(last_hidden_state)

        logits = self.classifier(last_hidden_state)
        
        return logits 
from transformers import BertModel
import torch.nn as nn
import torch

class BertNER(nn.Module):
    def __init__(self, num_classes):
        super(BertNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        predictions = self.linear(embeddings)

        return predictions


class BertNERBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(BertNERBiLSTM, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, bidirectional=True)
        self.linear = nn.Linear(self.embedding_dim * 2, self.num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        predictions = self.lstm(embeddings)[0]
        predictions = self.dropout(predictions)
        predictions = self.linear(predictions)

        return predictions


class ContextBertNER(nn.Module):
    def __init__(self, num_classes, device):
        super(ContextBertNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.device = device

        self.bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.embedding_dim, bidirectional=True)
        self.linear = nn.Linear(self.embedding_dim * 2, self.num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        last_hidden_state = self.bert(batch)[0]

        result_hidden_state = last_hidden_state.clone()
        result_hidden_state = result_hidden_state.detach().cpu()

        additional_context = torch.zeros_like(result_hidden_state, requires_grad=False)

        token2embedding = {}

        for batch_element_id, tokens in enumerate(batch):
            for token_id, token in enumerate(tokens):
                token = token.item()
                if token in token2embedding.keys():
                    token2embedding[token].append(result_hidden_state[batch_element_id][token_id])
                else:
                    token2embedding[token] = [result_hidden_state[batch_element_id][token_id]]

        for token in token2embedding.keys():
            token2embedding[token] = torch.mean(torch.stack(token2embedding[token]), dim=0)

        for batch_element_id, tokens in enumerate(batch):
            for token_id, token in enumerate(tokens):
                token = token.item()
                additional_context[batch_element_id][token_id] = token2embedding[token]

        additional_context = additional_context.to(self.device)
        hidden_state_with_context = torch.cat((last_hidden_state, additional_context), 2)

        predictions = self.lstm(hidden_state_with_context)[0]
        predictions = self.dropout(predictions)
        predictions = self.linear(predictions)

        return predictions


class DocumentContextBertNER(nn.Module):
    def __init__(self, num_classes, device):
        super(DocumentContextBertNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.device = device

        self.bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.embedding_dim, bidirectional=True)
        self.linear = nn.Linear(self.embedding_dim * 2, self.num_classes)
        self.dropout = nn.Dropout(0.1)

    def get_document_context(self, document):
        last_hidden_state = self.bert(document)[0]

        result_hidden_state = last_hidden_state.clone()
        result_hidden_state = result_hidden_state.detach().cpu()

        token2embedding = {}

        for document_element_id, tokens in enumerate(document):
            for token_id, token in enumerate(tokens):
                token = token.item()
                if token in token2embedding.keys():
                    token2embedding[token].append(result_hidden_state[document_element_id][token_id])
                else:
                    token2embedding[token] = [result_hidden_state[document_element_id][token_id]]

        for token in token2embedding.keys():
            token2embedding[token] = torch.mean(torch.stack(token2embedding[token]), dim=0)

        return token2embedding

    def forward(self, batch, documents_ids, mean_embeddings):
        last_hidden_state = self.bert(batch)[0]

        result_hidden_state = last_hidden_state.clone()
        result_hidden_state = result_hidden_state.detach().cpu()

        additional_context = torch.zeros_like(result_hidden_state, requires_grad=False)

        for batch_element_id, tokens in enumerate(batch):
            document_id = documents_ids[batch_element_id]
            for token_id, token in enumerate(tokens):
                token = token.item()
                additional_context[batch_element_id][token_id] = mean_embeddings[document_id][token]

        additional_context = additional_context.to(self.device)
        hidden_state_with_context = torch.cat((last_hidden_state, additional_context), 2)

        predictions = self.lstm(hidden_state_with_context)[0]
        predictions = self.dropout(predictions)
        predictions = self.linear(predictions)

        return predictions
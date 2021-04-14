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


class BertNERWithFirstContext(nn.Module):
    def __init__(self, num_classes, device):
        super(BertNERWithFirstContext, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.token2embedding = {}
        self.device = device

        self.bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, batch):
        last_hidden_state = self.bert(batch)[0]

        result_hidden_state = last_hidden_state.clone()
        result_hidden_state = result_hidden_state.detach().cpu()

        for batch_element_id, tokens in enumerate(batch):
            for token_id, token in enumerate(tokens):
                token = token.item()
                if token in self.token2embedding.keys():
                    result_hidden_state[batch_element_id][token_id] = self.token2embedding[token]
                else:
                    self.token2embedding[token] = result_hidden_state[batch_element_id][token_id]

        last_hidden_state = torch.Tensor(result_hidden_state).to(self.device).clone()
        predictions = self.linear(last_hidden_state)

        return predictions


class BertNERWithPreviousContextMean(nn.Module):
    def __init__(self, num_classes, device):
        super(BertNERWithPreviousContextMean, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.token2embedding = {}
        self.device = device

        self.bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, batch):
        last_hidden_state = self.bert(batch)[0]

        result_hidden_state = last_hidden_state.clone()
        result_hidden_state = result_hidden_state.detach().cpu()

        for batch_element_id, tokens in enumerate(batch):
            for token_id, token in enumerate(tokens):
                token = token.item()
                if token in self.token2embedding.keys():
                    self.token2embedding[token] = (self.token2embedding[token] + result_hidden_state[batch_element_id][
                        token_id]) / 2
                    result_hidden_state[batch_element_id][token_id] = self.token2embedding[token]
                else:
                    self.token2embedding[token] = result_hidden_state[batch_element_id][token_id]

        last_hidden_state = torch.Tensor(result_hidden_state).to(self.device).clone()
        predictions = self.linear(last_hidden_state)

        return predictions


class MeanLastStateBertNERwithoutBERTtags(nn.Module):
    def __init__(self, num_classes, device):
        super(MeanLastStateBertNERwithoutBERTtags, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.device = device

        self.bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.linear = nn.Linear(self.embedding_dim * 2, self.num_classes)

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
            token2embedding[token] = torch.mean(torch.stack(token2embedding[token]))

        for batch_element_id, tokens in enumerate(batch):
            for token_id, token in enumerate(tokens):
                token = token.item()
                additional_context[batch_element_id][token_id] = token2embedding[token]

        additional_context = additional_context.to(self.device)
        hidden_state_with_context = torch.cat((last_hidden_state, additional_context), 2)

        predictions = self.linear(hidden_state_with_context)

        return predictions
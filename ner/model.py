from transformers import BertModel, T5EncoderModel
import torch.nn as nn
import torch


class BERT(nn.Module):
    def __init__(self, model_name: str, classes: int, dropout_value: float = 0.1,
                 use_lstm: bool = False, lstm_layers: int = 1, lstm_size: int = 768):
        super(BERT, self).__init__()
        self.model_name = model_name

        if self.model_name == 'bert-base-cased':
            self.embedding_dim = 768
        elif self.model_name == 'bert-large-cased':
            self.embedding_dim = 1024
        else:
            raise ValueError('Model name is not valid.')

        self.classes = classes
        self.dropout_value = dropout_value
        self.use_lstm = use_lstm
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_size

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_size // 2,
                            bidirectional=True, num_layers=self.lstm_layers)
        self.linear = nn.Linear(self.embedding_dim, self.classes)
        self.linear_lstm = nn.Linear(self.lstm_hidden_size, self.classes)
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, input_ids, attention_mask=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        if self.use_lstm:
            predictions = self.lstm(last_hidden_state)[0]
            predictions = self.dropout(predictions)
            predictions = self.linear_lstm(predictions)
        else:
            predictions = self.dropout(last_hidden_state)
            predictions = self.linear(predictions)

        return predictions


class T5(nn.Module):
    def __init__(self, model_name: str, classes: int):
        super(T5, self).__init__()

        self.model_name = model_name
        if self.model_name == 't5-small':
            self.embedding_dim = 512
        elif self.model_name == 't5-base':
            self.embedding_dim = 768
        elif self.model_name == 't5-large':
            self.embedding_dim = 1024
        else:
            raise ValueError('Model name is not valid..')

        self.classes = classes

        self.t5 = T5EncoderModel.from_pretrained(model_name, output_attentions=True)
        self.linear = nn.Linear(self.embedding_dim, self.classes)

    def forward(self, input_ids, attention_masks=None):
        last_hidden_state = self.t5(input_ids=input_ids, attention_mask=attention_masks)[0]
        predictions = self.linear(last_hidden_state)

        return predictions


class BatchBPEContextBertNER(nn.Module):
    def __init__(self, num_classes, device):
        super(BatchBPEContextBertNER, self).__init__()
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


class DocumentBPEContextBertNER(nn.Module):
    def __init__(self, num_classes, device):
        super(DocumentBPEContextBertNER, self).__init__()
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


class DocumentContextBERT(nn.Module):
    def __init__(self, model_name: str, classes: int, dropout_value: float = 0.1, allow_flow_grad: bool = False,
                 use_lstm: bool = False, lstm_layers: int = 1, lstm_size: int = 768, device=torch.device('cpu')):
        super(DocumentContextBERT, self).__init__()
        self.model_name = model_name

        if self.model_name == 'bert-base-cased':
            self.embedding_dim = 768
        elif self.model_name == 'bert-large-cased':
            self.embedding_dim = 1024
        else:
            raise ValueError('Model name is not specified.')

        self.classes = classes
        self.dropout_value = dropout_value
        self.allow_flow_grad = allow_flow_grad

        self.use_lstm = use_lstm
        self.device = device
        self.lstm_hidden_size = lstm_size
        self.lstm_layers = lstm_layers

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.lstm_hidden_size // 2,
                            bidirectional=True, num_layers=self.lstm_layers)
        self.linear = nn.Linear(self.embedding_dim * 2, self.classes)
        self.linear_lstm = nn.Linear(self.lstm_hidden_size, self.classes)
        self.dropout = nn.Dropout(self.dropout_value)

    def get_document_context(self, document, words):
        last_hidden_state = self.bert(document)[0]

        for key in words:
            current_word = []
            for pos in words[key]['pos']:
                sentence_id = pos['sentence_id']
                if len(pos['ids']) == 1:
                    position = pos['ids']
                    current_word.append(last_hidden_state[sentence_id][position])
                else:
                    position_start = pos['ids'][0]
                    position_end = pos['ids'][-1]
                    current_word.append(last_hidden_state[sentence_id][position_start: position_end + 1])

            all_context_vectors_of_a_word = torch.stack(current_word, dim=0)
            mean_context_vector_of_a_word = torch.mean(all_context_vectors_of_a_word, dim=0)

            words[key]['context_vector'] = mean_context_vector_of_a_word

        return words

    def forward(self, batch, attention_masks, documents_ids, sentences_ids, mean_embeddings_for_batch_documents,
                sentences_from_documents):
        last_hidden_state = self.bert(input_ids=batch, attention_mask=attention_masks)[0]
        if self.allow_flow_grad:
            additional_context = last_hidden_state.clone()
        else:
            additional_context = last_hidden_state.clone().detach()
            additional_context.requires_grad_(requires_grad=False)

        for batch_element_id, tokens in enumerate(batch):
            document_id = documents_ids[batch_element_id]
            sentence_id = sentences_ids[batch_element_id]

            words_from_sentences = sentences_from_documents[document_id][sentence_id]
            words_from_document = mean_embeddings_for_batch_documents[document_id]

            for word in words_from_sentences:
                word_bpe = words_from_sentences[word]['bpe']

                once_seen = False
                for key in words_from_document:
                    if words_from_document[key]['bpe'] == word_bpe:
                        if len(words_from_document[key]['pos']) == 1:
                            once_seen = True
                        else:
                            context_vector = words_from_document[key]['context_vector']
                        break

                if once_seen == True:
                    pass
                else:
                    word_positions = words_from_sentences[word]['positions']

                    if word_bpe != '[PAD]':
                        if len(word_positions) == 1:
                            position = word_positions[0]
                            additional_context[batch_element_id][position] = context_vector
                        else:
                            for bpe_token_relative_pos, position_in_sentence in enumerate(word_positions):
                                additional_context[batch_element_id][position_in_sentence] = context_vector[
                                    bpe_token_relative_pos]
                    else:
                        for idx in range(word_positions[0], len(tokens)):
                            additional_context[batch_element_id][idx] = context_vector

                        for key in words_from_document:
                            if words_from_document[key]['bpe'] == ['[SEP]']:
                                context_vector = words_from_document[key]['context_vector']
                                additional_context[batch_element_id][-1] = context_vector
                                break

                    break

        additional_context = additional_context.to(self.device)
        hidden_state_with_context = torch.cat((last_hidden_state, additional_context), 2)

        if self.use_lstm:
            predictions = self.lstm(hidden_state_with_context)[0]
            predictions = self.dropout(predictions)
            predictions = self.linear_lstm(predictions)
        else:
            predictions = self.dropout(hidden_state_with_context)
            predictions = self.linear(predictions)

        return predictions
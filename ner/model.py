from ner.big_bird_attention import BigBirdAttention

import torch
import torch.nn as nn

from transformers import BertModel, T5EncoderModel, BigBirdConfig
from nltk.corpus import stopwords

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

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)[0]

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
    def __init__(self, model_name: str, classes: int, big_bird_config=None, dropout_value: float = 0.1,
                 use_lstm: bool = False, lstm_layers: int = 1, lstm_size: int = 768):
        super(LongAttentionBERT, self).__init__()
        self.model_name = model_name
        self.big_bird_config = big_bird_config
        self.output = BertSelfOutput()

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

        self.sparse_attention = BigBirdAttention(config=self.big_bird_config)

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_size // 2,
                            bidirectional=True, num_layers=self.lstm_layers)
        self.linear = nn.Linear(self.embedding_dim, self.classes)
        self.linear_lstm = nn.Linear(self.lstm_hidden_size, self.classes)
        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)[2]

        if self.use_lstm:
            predictions = self.lstm(hidden_states[-1])[0]
            predictions = self.dropout(predictions)
            predictions = self.linear_lstm(predictions)
        else:
            # выравниваем каждый слой в hidden_states в 1 ряд [num_layers, batch_size * seqlen, embedding_dim]
            hidden_state_one_row = torch.stack(
                [torch.cat(torch.tensor_split(hidden_state, sections=hidden_state.shape[0]), dim=1).squeeze() for
                 hidden_state in hidden_states])
            attention_mask_one_row = torch.cat(torch.tensor_split(attention_mask, sections=attention_mask.shape[0]),
                                               dim=1).squeeze()
            # last_hidden_state.shape[0] вместо BatchSize, потому что последний батч может быть остатком от деления и != BatchSize
            attention_output = self.sparse_attention(hidden_state_one_row)[0]  # , attention_mask=attention_mask_one_row)[0]
            # на выходе получаем [num_layers, batch_size * seqlen, embedding_dim], берем последний слой
            # размера [batch_size * seqlen, embedding_dim] и превращаем его в [1, batch_size * seqlen, embedding_dim]
            output = self.output(attention_output)[-1].unsqueeze(0)
            # превращаем этот hidden_state в размерность батча [batch_size, seq_len, embedding_dim]
            batch_output = torch.cat(torch.tensor_split(output, sections=hidden_states[-1].shape[0], dim=1), dim=0)
            predictions = self.linear(batch_output)

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

        self.stopwords = set(stopwords.words('english'))

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.lstm_hidden_size // 2,
                            bidirectional=True, num_layers=self.lstm_layers)
        self.linear = nn.Linear(self.embedding_dim * 2, self.classes)
        self.linear_lstm = nn.Linear(self.lstm_hidden_size, self.classes)
        self.layer_norm = nn.LayerNorm([self.embedding_dim * 2])
        self.dropout = nn.Dropout(self.dropout_value)

    def get_document_context(self, document, document_words):
        """
        Считаем средние контекстные вектора для каждого слова в документе

        params: document - токенизированный документ, токены которого переведены в token_ids
        размер документа (количество предложений в документе, максимальная длина последовательности среди всех
        предложений)

        params: document_words - словарь вида
        {word_id (ключ): {bpe: [список WordPiece токенов слова], pos: [список словарей вида {ids: [int], sentence_id: int}}

        word_id - уникальный номер слова
        ids - позиции токенов слова в предложении, начиная от 0
        sentence_id - номер предложения в документе

        т.е связка ids - sentence_id дает информацию о позиции токенов конкретного слова в указанном предложении
        """

        # посылаем в BERT целый документ
        last_hidden_state = self.bert(document)[0]

        # проходим по каждому слову в документе
        for key in document_words:
            context_vectors_for_current_word = []
            # перебираем все вхождения слова в документ
            for positions in document_words[key]['pos']:
                sentence_id = positions['sentence_id']
                # если слово состоит из одного токена, то в позициях будет только один элемент
                if len(positions['ids']) == 1:
                    position = positions['ids']
                    context_vectors_for_current_word.append(last_hidden_state[sentence_id][position])
                # если слово состоит из нескольких токенов, то берем позиции первого токена и последнего, и берем slice
                else:
                    position_start = positions['ids'][0]
                    position_end = positions['ids'][-1]
                    context_vectors_for_current_word.append(
                        last_hidden_state[sentence_id][position_start: position_end + 1])

            # делаем тензор из списка контекстных векторов для слова
            all_context_vectors_of_a_word = torch.stack(context_vectors_for_current_word, dim=0)
            # берем среднее по каждому WordPiece токену
            mean_context_vector_of_a_word = torch.mean(all_context_vectors_of_a_word, dim=0)

            # кладем в изначальный словарь средний контекстный вектор для каждого токена конкретного слова
            document_words[key]['context_vector'] = mean_context_vector_of_a_word

        return document_words

    def forward(self, batch, attention_masks, documents_ids, sentences_ids, mean_embeddings_for_batch_documents,
                word_positions):
        last_hidden_state = self.bert(input_ids=batch, attention_mask=attention_masks)[0]

        additional_context = torch.zeros_like(last_hidden_state).detach()
        additional_context.requires_grad_(requires_grad=False)

        for batch_element_id, tokens in enumerate(batch):
            # для примера батча получаем номер его документа и номер позиции предложения внутри него
            document_id = documents_ids[batch_element_id]
            sentence_id = sentences_ids[batch_element_id]

            # берем слова из конкретного предложения и их позиции в токенах:
            # из словаря вида {document_id: {sentence_id: {word_id: {bpe: [], positions: []}}}}
            # пример:
            """
            {0: {0: {0: {'bpe': ['[CLS]'], 'positions': [0]},
                     1: {'bpe': ['EU'], 'positions': [1]},
                     2: {'bpe': ['rejects'], 'positions': [2]},
            """

            words_from_sentence = word_positions[document_id][sentence_id]

            # получаем словарь слов с посчитанными усредненными контекстными векторами для каждого слова в документе
            words_embeddings = mean_embeddings_for_batch_documents[document_id]

            for word in words_from_sentence:
                word_bpe = words_from_sentence[word]['bpe']

                # изначально считаем, что слово встречается не один раз
                for key in words_embeddings:
                    # нашли слово совпадению WordPiece токенов
                    if words_embeddings[key]['bpe'] == word_bpe:
                        # если слово встречается один раз, то мы не берем его средний вектор по документу
                        if len(words_embeddings[key]['pos']) == 1 or ''.join(word_bpe).replace('##', '') in self.stopwords:
                            context_vector = torch.zeros(768)
                        # если более одного раза, то будем менять, поэтому берем его средний вектор
                        else:
                            context_vector = words_embeddings[key]['context_vector']
                        # ранний выход из цикла, если мы нашли текущее слово
                        break

                # если слово встречается один раз в документе, то ничего не меняем в last_hidden_state по батчу
                word_positions_in_sentence = words_from_sentence[word]['positions']

                # если слово не является PAD токеном
                if word_bpe != ['[PAD]'] or word_bpe != ['SEP'] or word_bpe != ['CLS']:
                    # если слово из одного токена
                    if len(word_positions_in_sentence) == 1:
                        position = word_positions_in_sentence[0]
                        additional_context[batch_element_id][position] = context_vector
                    # если слово из нескольких токенов
                    else:
                        for bpe_token_relative_pos, position_in_sentence in enumerate(word_positions_in_sentence):
                            additional_context[batch_element_id][position_in_sentence] = context_vector[
                                bpe_token_relative_pos]

        # конкатенируем скрытый слой с батча и тензор с заменой на средние вектора
        hidden_state_with_context = torch.cat((last_hidden_state, additional_context), 2)

        if self.use_lstm:
            predictions = self.lstm(hidden_state_with_context)[0]
            predictions = self.dropout(predictions)
            predictions = self.linear_lstm(predictions)
        else:
            predictions = self.layer_norm(hidden_state_with_context)
            predictions = self.dropout(predictions)
            predictions = self.linear(predictions)

        return predictions
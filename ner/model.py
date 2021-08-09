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

    def get_document_context(self, document, document_words):
        """
        Считаем контекстные вектора для каждого слова в документе

        params: document - токенизированный документ, токены которого переведены в token_ids
        размер документа (количество предложений в документе, максимальная длина последовательности среди всех
        предложений)

        params: document_words - словарь вида
        {Word Id (ключ): {tokens: [список WordPiece токенов слова],
                          pos: [список словарей, где ключ - номер предлжения (в документе),
                                                 а значение - позиция его WordPiece токенов в этом предложении]}
        }
        Word Id - уникальный номер по документу
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
                    context_vectors_for_current_word.append(last_hidden_state[sentence_id][position_start: position_end + 1])

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

        # по умолчанию отключаем градиент у копируемого скрытого слоя (если allow_flow_grad=True, то не отключаем)
        if self.allow_flow_grad:
            additional_context = last_hidden_state.clone()
        else:
            additional_context = last_hidden_state.clone().detach()
            additional_context.requires_grad_(requires_grad=False)

        for batch_element_id, tokens in enumerate(batch):
            # для примера батча получаем номер его документа и номер позиции предложения внутри него
            document_id = documents_ids[batch_element_id]
            sentence_id = sentences_ids[batch_element_id]

            # для предложения получаем список вида: {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [6], ...},
            # где ключ - номер слова в предложений, значение - позиции его WordPiece токенов в предложении
            words_from_sentence = word_positions[document_id][sentence_id]
            # получаем словарь слов с посчитанными усредненными контекстными векторами для каждого слова в документе
            words_embeddings = mean_embeddings_for_batch_documents[document_id]

            for word in words_from_sentence:
                word_bpe = words_from_sentence[word]['bpe']

                # изначально считаем, что слово встречается не один раз
                once_seen = False
                for key in words_embeddings:
                    # нашли слово совпадению WordPiece токенов
                    if words_embeddings[key]['bpe'] == word_bpe:
                        # если слово встречается один раз, то мы не берем его средний вектор по документу
                        if len(words_embeddings[key]['pos']) == 1:
                            once_seen = True
                        # если более одного раза, то будем менять, поэтому берем его средний вектор
                        else:
                            context_vector = words_embeddings[key]['context_vector']
                        # ранний выход из цикла, если мы нашли текущее слово
                        break

                # если слово встречается один раз в документе, то ничего не меняем в last_hidden_state по батчу
                if once_seen == True:
                    pass
                else:
                    word_positions = words_from_sentence[word]['positions']

                    # если слово не является PAD токеном
                    if word_bpe != '[PAD]':
                        # если слово из одного токена
                        if len(word_positions) == 1:
                            position = word_positions[0]
                            additional_context[batch_element_id][position] = context_vector
                        # если слово из нескольких токенов
                        else:
                            for bpe_token_relative_pos, position_in_sentence in enumerate(word_positions):
                                additional_context[batch_element_id][position_in_sentence] = context_vector[
                                    bpe_token_relative_pos]
                    else:
                        # если это PAD токен, то берем его первое вхождение в предложение и проставляем его
                        # средний вектор до конца последовательности
                        for idx in range(word_positions[0], len(tokens)):
                            additional_context[batch_element_id][idx] = context_vector

                        # меняем вектор для SEP токена
                        for key in words_embeddings:
                            if words_embeddings[key]['bpe'] == ['[SEP]']:
                                context_vector = words_embeddings[key]['context_vector']
                                additional_context[batch_element_id][-1] = context_vector
                                break

                    break

        additional_context = additional_context.to(self.device)
        # конкатенируем скрытый слой с батча и тензор с заменой на средние вектора
        hidden_state_with_context = torch.cat((last_hidden_state, additional_context), 2)

        if self.use_lstm:
            predictions = self.lstm(hidden_state_with_context)[0]
            predictions = self.dropout(predictions)
            predictions = self.linear_lstm(predictions)
        else:
            predictions = self.dropout(hidden_state_with_context)
            predictions = self.linear(predictions)

        return predictions
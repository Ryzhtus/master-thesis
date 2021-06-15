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


class DocumentWordContextBertNER(nn.Module):
    def __init__(self, num_classes, device):
        super(DocumentWordContextBertNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.device = device

        self.bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.lstm = nn.LSTM(self.embedding_dim * 2, self.embedding_dim, bidirectional=True)
        self.linear = nn.Linear(self.embedding_dim * 2, self.num_classes)
        self.dropout = nn.Dropout(0.1)

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

    def forward(self, batch, documents_ids, sentences_ids, mean_embeddings_for_batch_documents,
                sentences_from_documents):
        last_hidden_state = self.bert(batch)[0]
        additional_context = last_hidden_state.clone()

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

        predictions = self.lstm(hidden_state_with_context)[0]
        predictions = self.dropout(predictions)
        predictions = self.linear(predictions)

        return predictions
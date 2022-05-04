import torch 

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from reader import Document

class ChunksDataset(Dataset):
    def __init__(self, documents: list[Document], max_sequence_length: int, tokenizer: PreTrainedTokenizer):
        self.documents = documents
        self.sentences = [sentence for document in self.documents for sentence in document]

        self.max_sequence_length = max_sequence_length

        self.empty_word_vector = None
        
        self.entity_tags = sorted(list(set(word.label for sentence in self.sentences for word in sentence)))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.entity_tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.entity_tags)}

        self.tokenizer = tokenizer

        # превращаем предложения в чанки (режем по длине)
        #self.chunk_reindex()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]

        tokenized_words = []
        tokenized_labels = []
        tokenized_words_features = []
        words_ids = [] # индексы начала слов в субтокенах

        for word in sentence:
            subtokens = self.tokenizer.tokenize(word.token)
        
            for idx, _ in enumerate(subtokens):
                if idx == 0:
                    tokenized_labels.append(self.tag2idx[word.label])
                else:
                    tokenized_labels.append(-100)

                tokenized_words_features.append(word.feature_vector())

            # добавляем новый индекс для следующего слова
            words_ids.append(len(tokenized_words) + 1)
            # добавляем субтокены/токены в общий список
            tokenized_words.extend(subtokens)

        assert len(tokenized_words) == len(tokenized_words_features)

        # добавляем [CLS] и [SEP] токены и конвертируем их в ids
        tokens = [self.tokenizer.cls_token] + tokenized_words + [self.tokenizer.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # создаем пустой вектор признаков слова, который присвоим [CLS], [SEP] и [PAD] токенам
        empty_word_feature_vector = [0 for _ in range(len(tokenized_words_features[0]))]
        self.empty_word_vector = empty_word_feature_vector
        
        # добавляем лейблы -100 и пустой вектор признаков для [CLS] и [SEP] токенов
        label_ids = [-100] + tokenized_labels + [-100]
        tokenized_words_features = [empty_word_feature_vector] + tokenized_words_features + [empty_word_feature_vector]
        
        attention_mask = [1] * len(tokens_ids)
        wordpiece_mask = [1 if label_item != -100 else -100 for label_item in label_ids]

        return torch.LongTensor(tokens_ids), torch.LongTensor(label_ids), torch.LongTensor(attention_mask), torch.tensor(tokenized_words_features, dtype=torch.float), \
            torch.tensor(wordpiece_mask, dtype=torch.long) 
            

    # пока что depreceated метод
    def __chunk_reindex(self):
        chunk_id = 0  # общий номер чанка в датасете 
        chunk_local_id = 0  # номер чанка в документе
        previous_document_id = 0  # номер документа

        for sentence_index, (sentence, labels) in enumerate(zip(self.sentences, self.labels)):
            current_document_id = self.sentence2document[sentence_index]

            if current_document_id != previous_document_id:
                chunk_local_id = 0

            words = sentence
            words_labels = labels

            # проверяем меньше ли длина последовательности после токенизации, чем максимально допустимая
            # с учетом CLS и SEP токенов
            if len(self.tokenizer(words, is_split_into_words=True)['input_ids']) <= self.max_sequence_length:
                self.chunks.append(words)
                self.chunks_labels.append(words_labels)

                # заносим индексы в словари
                self.chunk2document[chunk_id] = current_document_id
                self.document2chunk[current_document_id].append(chunk_id)
                self.chunk2position[chunk_id] = {'document_id': current_document_id, 'sentence_pos_id': chunk_local_id}

                # не забываем прибавить 1 к счетчикам
                chunk_id += 1
                chunk_local_id += 1
            else:
                # разделяем предложение напополам (в словах)
                words_left = words[0: len(words) // 2]
                words_right = words[len(words) // 2: len(words)]

                words_labels_left = words_labels[0: len(words_labels) // 2]
                words_labels_right = words_labels[len(words_labels) // 2: len(words_labels)]

                self.chunks.append(words_left)
                self.chunks_labels.append(words_labels_left)

                # заносим индексы в словари
                self.chunk2document[chunk_id] = current_document_id
                self.document2chunk[current_document_id].append(chunk_id)
                self.chunk2position[chunk_id] = {'document_id': current_document_id, 'sentence_pos_id': chunk_local_id}

                # не забываем прибавить 1 к счетчикам
                chunk_id += 1
                chunk_local_id += 1

                self.chunks.append(words_right)
                self.chunks_labels.append(words_labels_right)

                # заносим индексы в словари
                self.chunk2document[chunk_id] = current_document_id
                self.document2chunk[current_document_id].append(chunk_id)
                self.chunk2position[chunk_id] = {'document_id': current_document_id, 'sentence_pos_id': chunk_local_id}

                # не забываем прибавить 1 к счетчикам
                chunk_id += 1
                chunk_local_id += 1

    def paddings(self, batch):
        input_ids, labels, attention_masks, features, wordpiece_mask = list(zip(*batch))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        features = pad_sequence(features, batch_first=True, padding_value=0)
        wordpiece_masks = pad_sequence(wordpiece_mask, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, 
                "labels": labels, 
                "attention_masks": attention_masks,
                "features": features, 
                "wordpiece_masks": wordpiece_masks}

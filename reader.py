import tqdm
import glob
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

@dataclass
class Word():
    def __init__(self, token, label) -> None:
        self.token = token
        self.label = label

        self.features = {
           # "capslock": False,
           # "capitalized": False,
            "tfidf": 0
           # "repeats": 0,
           # "repeats_upper": 0,
           # "repeats_lower": 0,
           # "always_with_next": False
           # "isdigit": False,
        }

    def __str__(self) -> str:
        return self.token

    def __repr__(self) -> str:
        return f"Word({self.token}, {self.label})"

    def __eq__(self, __o: object) -> bool:
        # compare if two Word() objects are equal. used for feature extraction
        return (self.token == __o.token) & (self.label == __o.label)

    def feature_vector(self) -> list[int]: 
        return [float(feature) for feature in self.features.values()]


@dataclass
class Span():
    def __init__(self) -> None:
        self.span = None
        self.label = None
        self.tokens = []
        self.labels = []

    def __repr__(self) -> str:
        return f"Span({self.span}, {self.label})"

    def __str__(self) -> None:
        return self.span
        
    def add(self, word: Word) -> None:
        if self.span:
            self.span += " "
            self.span += str(word.token)
        else:
            self.span = str(word.token)

        self.tokens.append(word.token)
        self.labels.append(word.label)

        if not self.label:
            if len(word.label) == 1:
                self.label = word.label
            else:
                self.label = word.label[2:]


@dataclass
class Sentence():
    def __init__(self, words: list[Word]) -> None:
        self.words = words
        self.text = ' '.join([word.token for word in self.words])
        self.document_id = None
        self.sentence_id = None
        self.spans = []

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index: int):
        return self.words[index]

    def __repr__(self) -> str:
        return str({"text": self.text,
                    "words": [str(word) for word in self.words],
                    "spans": [str(span) for span in self.spans if span.label != "O"]})

    def __str__(self) -> str:
        return self.text

    def __iter__(self):
        return iter(self.words)

    def __bioes(self) -> None:
        # converts IOB2 into IOBES         
        for idx, word in enumerate(self.words):
            if word.label == "O":
                continue
            elif word.label.split('-')[0] == 'B':
                if idx + 1 != len(self.words) and self.words[idx + 1].label.split('-')[0] == 'I':
                    continue
                else:
                    word.label = word.label.replace('B-', 'S-')
            elif word.label.split('-')[0] == 'I':
                if idx + 1 < len(self.words) and self.words[idx + 1].label.split('-')[0] == 'I':
                    continue
                else:
                    word.label = word.label.replace('I-', 'E-')
            else:
                raise Exception('Invalid IOB format!')

    def __bioes_spans(self) -> None:
        # find IOBES spans in sentence
        for idx in range(len(self.words)):
            word = self.words[idx]

            if word.label == 'O':
                span = Span()
                span.add(word)
                self.__add_span(span)
            elif word.label[0] == "S":
                span = Span()
                span.add(word)
                self.__add_span(span)
            elif word.label[0] == "B":
                span = Span()
                span.add(word)
            elif word.label[0] == "I":
                span.add(word)
            elif word.label[0] == "E":
                span.add(word)
                self.__add_span(span)

    def __bio_spans(self) -> None:
        # find IOB spans in sentence 
        for idx in range(len(self.words) - 1):
            word = self.words[idx]

            if word.label == 'O':
                span = Span()
                span.add(word)
                self.__add_span(span)
            elif word.label[0] == "B":
                span = Span()
                span.add(word)
                if self.words[idx + 1].label == "O":
                    self.__add_span(span)
            elif word.label[0] == "I":
                if self.words[idx + 1].label == "O":
                    span.add(word)
                    self.__add_span(span)
                else:
                    span.add(word)

        if self.words[-1].label == 'O':
            span = Span()
            span.add(self.words[-1])
            self.__add_span(span)
        elif self.words[-1].label[0] == 'B':
            span = Span()
            span.add(self.words[-1])
            self.__add_span(span)
        elif self.words[-1].label[0] == 'I':
            span.add(self.words[-1])
            self.__add_span(span)

    def __add_span(self, span: Span) -> None:
        # add span to the list of sentence's spans
        self.spans.append(span)

    def bioes(self) -> None:
        self.__bioes()

@dataclass
class Document():
    def __init__(self, sentences: list[Sentence], document_id: int) -> None:
        self.sentences = sentences
        self.document_id = document_id

        self.document_text = " ".join([sentence.text for sentence in self.sentences])

        for sentence_id, sentence in enumerate(self.sentences):
            sentence.bioes()
            sentence.sentence_id = sentence_id
            sentence.document_id = self.document_id

        # self.__collect_capitalization_features()

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences)

    def __collect_capitalization_features(self) -> None:
        for sentence in self.sentences:
            for idx, word in enumerate(sentence.words):                
                if word.token[0].isupper():
                    if word.token.isupper():
                        word.features['capslock'] = True
                    else:
                        word.features["capitalized"] = True

    def __collect_occurances_features(self) -> None:
        # collect features for all words in a document
        for sentence in self.sentences:
            for idx, word in enumerate(sentence.words):                
                if word.token[0].isupper():
                    for sentence_re in self.sentences:
                        for word_re in sentence_re.words:
                            if word.token.casefold() == word_re.token.casefold():
                                word.features["repeats"] += 1

                                if word_re.token.isupper():
                                    word.features["repeats_upper"] += 1
                               
                                if word_re.token.islower():
                                    word.features["repeats_lower"] += 1
                    
                    # считаем признак совместной встречаемости слов
                    if idx != len(sentence) - 1:
                        number_of_coocurence = 0
                        next_word = sentence.words[idx + 1].token
                        words = word.token + " " + next_word
                        for sentence_re in self.sentences:
                            for word_idx in range(len(sentence_re.words) - 1):
                                word_re1 = sentence_re.words[word_idx].token
                                word_re2 = sentence_re.words[word_idx].token
                                words_re = word_re1 + " " + word_re2
                                if words.casefold() == words_re.casefold():
                                    number_of_coocurence += 1
                        
                        if number_of_coocurence != 0:
                            if number_of_coocurence == word.features["repeats"]:
                                word.features["always_with_next"] = True

                    div_factor = min(10, word.features["repeats"])

                    word.features["repeats_upper"] /= word.features["repeats"]
                    word.features["repeats_lower"] /= word.features["repeats"]        
                    word.features["repeats"] /= div_factor  

class ReaderCoNLL():
    def __init__(self, tokenizer: BertTokenizer, max_length_in_tokens: int = 128) -> None:
        self.documents = []
        self.vectorizer = TfidfVectorizer(lowercase=False, stop_words="english")
        self.fit_vectorizer = False
        self.tokenizer = tokenizer
        self.max_length_in_tokens = max_length_in_tokens

    def __parse(self, filename: str):
        rows = open(filename, 'r').read().strip().split("\n\n")
        sentences = []
        documents = []
        document_sentence = []

        document_id = -1

        for sentence in tqdm.tqdm(rows):       
            for line in sentence.splitlines():
                token = line.split()[0]
                label = line.split()[-1]       
                word = Word(token, label)

                if token == "-DOCSTART-":
                    if document_id == -1:
                        document_id += 1
                    else:
                        self.documents.append(Document(sentences, document_id))
                        document_id += 1
                        sentences = []
                    
                document_sentence.append(word)
            
            sentence_text = " ".join([word.token for word in document_sentence])
            tokenized_text = self.tokenizer(sentence_text)
            
            if len(tokenized_text.input_ids) > self.max_length_in_tokens:
                pass

            sentences.append(Sentence(document_sentence))
            document_sentence = []

        # don't forget to add the last document (because there is no -DOCSTART- tag in the ned of a file)
        self.documents.append(Document(sentences, document_id))
        # self.__compute_tf_ifd()

        return self.documents

    def __compute_tf_ifd(self) -> None:
        document_texts = [document.document_text for document in self.documents]

        if self.fit_vectorizer:
            self.vectorizer.fit(document_texts)

        vocab = list(self.vectorizer.get_feature_names_out())

        for document in self.documents:
            doc = document.document_text
            doc_freq = self.vectorizer.transform([doc])

            for sentence in document:
                for word in sentence:
                    if word.token in vocab:
                        idx_in_vocab = vocab.index(word.token)
                        word.features["tfidf"] = float(doc_freq.getcol(idx_in_vocab).toarray()[0])
                    else:
                        word.features["tfidf"] = 0

    def parse(self, filename: str, fit_vectorizer: bool = False) -> list[Document]:
        self.documents = []
        self.fit_vectorizer = fit_vectorizer
        return self.__parse(filename)

class ReaderOntonotes():
    def __init__(self, tokenizer: BertTokenizer, max_length_in_tokens: int = 128) -> None:
        self.documents = []
        self.vectorizer = TfidfVectorizer(lowercase=False, stop_words="english")
        self.fit_vectorizer = False
        self.tokenizer = tokenizer
        self.max_length_in_tokens = max_length_in_tokens

    def __parse_document(self, filename: str, document_id: int) -> None:
        rows = open(filename, 'r').read().strip().split('\n')
        sentences = []
        document_sentence = []

        for row in rows:
            data = row.split(" ")
            
            # check if sentence is emtpy
            if data[0] == "":
                if document_sentence != []:
                    sentences.append(Sentence(document_sentence))
                    document_sentence = []
            else:
                token = data[0]
                label = data[1]
                word = Word(token, label)
                document_sentence.append(word)

        splitted_sentences = self.__split_sentence(sentences)
        self.documents.append(Document(splitted_sentences, document_id))

    def __split_sentence(self, original_sentences: list[Sentence]) -> list[Sentence]:
        splitted_sentences = []

        for sentence in original_sentences:
            tokenized_text = self.tokenizer._tokenizer.encode(sentence.text)

            if len(tokenized_text) > self.max_length_in_tokens:
                split_id = tokenized_text.token_to_word(self.max_length_in_tokens // 2)

                sentence_a = Sentence(sentence[:split_id])
                sentence_b = Sentence(sentence[split_id:])

                splitted_sentences.append(sentence_a)
                splitted_sentences.append(sentence_b)
            else:
                splitted_sentences.append(sentence)

        return splitted_sentences

    def __compute_tf_ifd(self) -> None:
        document_texts = [document.document_text for document in self.documents]

        if self.fit_vectorizer:
            self.vectorizer.fit(document_texts)

        vocab = list(self.vectorizer.get_feature_names_out())

        for document in tqdm.tqdm(self.documents):
            doc = document.document_text
            doc_freq = self.vectorizer.transform([doc])

            for sentence in document:
                for word in sentence:
                    if word.token in vocab:
                        idx_in_vocab = vocab.index(word.token)
                        word.features["tfidf"] = float(doc_freq.getcol(idx_in_vocab).toarray()[0])
                    else:
                        word.features["tfidf"] = 0

    def parse(self, foldername: str, fit_vectorizer: bool = False) -> list[Document]:
        self.fit_vectorizer = fit_vectorizer
        filenames = glob.glob(foldername + "\\" + "*.txt")
        self.documents = []

        for document_id, filename in tqdm.tqdm(enumerate(filenames)):
            self.__parse_document(filename, document_id)
        
        self.__compute_tf_ifd()

        return self.documents
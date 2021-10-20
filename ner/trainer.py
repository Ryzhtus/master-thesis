import comet_ml

from typing import List, Dict
from ner.metrics import FMeasureStorage
from ner.utils import clear_for_metrics
from seqeval.metrics import performance_measure, classification_report, f1_score
from tqdm import tqdm
from ner.document import Document
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim.optimizer


class Trainer():
    def __init__(self, experiment: comet_ml.Experiment, model, params: dict,
                 optimizer, criterion, scheduler, clip_grad: bool,
                 epochs: int, last_epoch: bool,
                 train_data: torch.utils.data.DataLoader,
                 eval_data: torch.utils.data.DataLoader,
                 test_data: torch.utils.data.DataLoader,
                 train_documents: Document,
                 eval_documents: Document,
                 test_documents: Document,
                 tag2idx: dict, idx2tag: dict,
                 device):
        self.model = model
        self.params = params

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_grad = clip_grad

        self.epochs = epochs
        self.last_epoch = last_epoch

        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.train_documents = train_documents
        self.eval_documents = eval_documents
        self.test_documents = test_documents

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

        self.device = device

        self.experiment = experiment
        self.experiment.log_parameters(self.params)

        self.train_loss = []
        self.eval_loss = []
        self.test_loss = []

        self.train_span_f1 = []
        self.eval_span_f1 = []

        self.epoch_labels = []
        self.epoch_predictions = []

        # variables for logging eval loss and test loss
        self.eval_steps = 0
        self.test_steps = 0

        self.progress_info = '{:>5s} Loss = {:.5f}, Token F1-score = {:.2%}, Span F1-score = {:.2%}'

    def __step(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
               words_ids: List[List[int]], document_ids: List[int] = None, sentences_ids: List[int] = None,
               document_word_embeddings: Dict = None, word_positions: Dict = None,
               freeze_bert: bool = False):
        """
        Общий метод, используемый для шага обучения, валидации и тестирования.
        В этом блоке модель делает предсказание по батчу, считается значение функции потерь и полученные предсказания
        сохраняются в списке, чтобы в конце эпохе честно посчитать метрики по сущносят
        """

        if freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        if document_ids and sentences_ids and document_word_embeddings and word_positions:
            predictions = self.model(input_ids, attention_mask, document_ids, sentences_ids, document_word_embeddings,
                                     word_positions)
        else:
            predictions = self.model(input_ids, attention_mask)

        loss = self.criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))

        predictions = predictions.argmax(dim=2).cpu().numpy()
        labels = labels.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_for_metrics(labels, predictions, self.idx2tag, words_ids)

        iteration_result = performance_measure(clear_labels, clear_predictions)

        self.epoch_labels += clear_labels
        self.epoch_predictions += clear_predictions

        return loss, iteration_result

    def __get_document_word_vectors(self, document_ids: List[int], documents: Document):
        """
        Под выключенным градиентом у BERT считаем для каждого документа средние вектора его слов, сохраняем их
        в словарь
        """
        with torch.no_grad():
            document_word_embeddings = {}
            # variable for each word's positions in each document in sentence order
            word_positions = {}

            for document_id in set(document_ids):
                document_word_embeddings[document_id] = self.model.get_document_context(
                    documents[document_id].to(self.device), documents.collect_all_positions_for_each_word(document_id))
                word_positions[document_id] = documents.get_document_words_by_sentences(document_id)

        return document_word_embeddings, word_positions

    def __train_epoch(self, name: str, freeze_bert=False):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()

        self.model.train()
        with self.experiment.train():
            with tqdm(total=len(self.train_data)) as progress_bar:
                for batch in self.train_data:
                    self.optimizer.zero_grad()

                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device)
                    words_ids = batch[3]

                    # если есть документы, то используем модель, учитывающую контекст документа
                    if self.train_documents:
                        document_ids = batch[4]
                        sentences_ids = batch[5]
                        document_word_embeddings, word_positions = self.__get_document_word_vectors(
                            document_ids, self.train_documents)
                        loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids, document_ids,
                                                    sentences_ids, document_word_embeddings,
                                                    word_positions, freeze_bert)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids, freeze_bert=freeze_bert)

                    epoch_metrics += step_f1
                    epoch_loss += loss.item()
                    self.train_loss.append(loss.item())

                    loss.backward()

                    if self.clip_grad:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                    self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data), 0, 0))

                progress_bar.update()
                epoch_token_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data),
                                                                       epoch_token_f1_score, epoch_span_f1_score))
                self.experiment.log_metric("Span F1", epoch_span_f1_score)

            self.train_span_f1.append(epoch_span_f1_score)

    def __eval_epoch(self, name):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()

        self.model.eval()
        with self.experiment.validate():
            with tqdm(total=len(self.eval_data)) as progress_bar:
                for batch in self.eval_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device)
                    words_ids = batch[3]

                    if self.eval_documents:
                        document_ids = batch[4]
                        sentences_ids = batch[5]
                        document_word_embeddings, word_positions = self.__get_document_word_vectors(
                            document_ids, self.eval_documents)

                        with torch.no_grad():
                            loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids, document_ids,
                                                        sentences_ids, document_word_embeddings,
                                                        word_positions)
                    else:
                        with torch.no_grad():
                            loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids)

                    epoch_metrics += step_f1
                    epoch_loss += loss.item()
                    self.eval_loss.append(loss.item())
                    self.experiment.log_metric('loss', loss.item(), step=self.eval_steps)
                    self.eval_steps += 1

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.eval_data),
                                                                           0, 0))

                progress_bar.update()
                epoch_token_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.eval_data),
                                                                       epoch_token_f1_score, epoch_span_f1_score))
                self.experiment.log_metric("Span F1", epoch_span_f1_score)

            self.eval_span_f1.append(epoch_span_f1_score)

    def __test_epoch(self, name: str):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()

        self.model.eval()
        with self.experiment.test():
            with tqdm(total=len(self.test_data)) as progress_bar:
                for batch in self.test_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device)
                    words_ids = batch[3]

                    if self.test_documents:
                        document_ids = batch[4]
                        sentences_ids = batch[5]
                        document_word_embeddings, word_positions = self.__get_document_word_vectors(
                            document_ids, self.test_documents)

                        with torch.no_grad():
                            loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids, document_ids,
                                                        sentences_ids, document_word_embeddings,
                                                        word_positions)
                    else:
                        with torch.no_grad():
                            loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids)

                    epoch_metrics += step_f1
                    epoch_loss += loss.item()
                    self.test_loss.append(loss.item())
                    self.experiment.log_metric('loss', loss.item(), step=self.test_steps)
                    self.test_steps += 1

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.test_data),
                                                                           0, 0))

                progress_bar.update()
                epoch_token_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.test_data),
                                                                       epoch_token_f1_score, epoch_span_f1_score))
                self.experiment.log_metric("Span F1", epoch_span_f1_score)

    def fit(self):
        for epoch in range(self.epochs):
            progress = '[{} / {}] '.format(epoch + 1, self.epochs)

            if epoch == (self.epochs - 1) and self.last_epoch:
                self.__train_epoch(progress + 'Train:', freeze_bert=True)
            else:
                self.__train_epoch(progress + 'Train:')

            # clear labels and predictions from memory after training epoch
            self.epoch_labels = []
            self.epoch_predictions = []

            self.__eval_epoch(progress + 'Eval :')

            # clear labels and predictions from memory after validation epoch
            self.epoch_labels = []
            self.epoch_predictions = []

    def test(self):
        self.__test_epoch('Test :')
        print(classification_report(self.epoch_labels, self.epoch_predictions, digits=4))

    def plot_loss_curve(self):
        plt.figure(figsize=(14, 7))
        plt.title('Значения функции потерь для модели {}'.format(type(self.model)))
        plt.plot([i for i in range(len(self.train_loss))], self.train_loss, label='Train')
        plt.plot([i for i in range(len(self.eval_loss))], self.eval_loss, label='Validation')
        plt.xticks([i for i in range(self.params['epochs'])])
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.show()

    def plot_score_curve(self):
        plt.figure(figsize=(14, 7))
        plt.title('Значения Span F1 для модели {}'.format(type(self.model)))
        plt.plot([i for i in range(self.params['epochs'])], self.train_span_f1, label='Train')
        plt.plot([i for i in range(self.params['epochs'])], self.eval_span_f1, label='Validation')
        plt.xticks([i for i in range(self.params['epochs'])])
        plt.xlabel('Epochs')
        plt.ylabel('Span F1 Value')
        plt.legend()
        plt.show()
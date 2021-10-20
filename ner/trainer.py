import comet_ml

from typing import List, Dict
from ner.utils import clear_for_metrics
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim.optimizer


class Trainer():
    def __init__(self, experiment: comet_ml.Experiment, model, params: dict,
                 optimizer, criterion, scheduler, clip_grad: bool, epochs: int,
                 train_data: torch.utils.data.DataLoader,
                 eval_data: torch.utils.data.DataLoader,
                 test_data: torch.utils.data.DataLoader,
                 tag2idx: dict, idx2tag: dict,
                 device):
        self.model = model
        self.params = params

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_grad = clip_grad

        self.epochs = epochs

        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

        self.device = device

        self.experiment = experiment
        self.experiment.log_parameters(self.params)

        self.epoch_labels = []
        self.epoch_predictions = []

        self.progress_info = '{:>5s} Loss = {:.5f}, Span F1-score = {:.2%}'

    def __step(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
               words_ids: List[List[int]]):
        """
        Общий метод, используемый для шага обучения, валидации и тестирования.
        В этом блоке модель делает предсказание по батчу, считается значение функции потерь и полученные предсказания
        сохраняются в списке, чтобы в конце эпохе честно посчитать метрики по сущносят
        """

        predictions = self.model(input_ids, attention_mask)

        loss = self.criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))

        predictions = predictions.argmax(dim=2).cpu().numpy()
        labels = labels.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_for_metrics(labels, predictions, self.idx2tag, words_ids)

        self.epoch_labels += clear_labels
        self.epoch_predictions += clear_predictions

        return loss

    def __train_epoch(self, name: str):
        epoch_loss = 0

        self.model.train()
        with self.experiment.train():
            with tqdm(total=len(self.train_data)) as progress_bar:
                for batch in self.train_data:
                    self.optimizer.zero_grad()

                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device)
                    words_ids = batch[3]

                    loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids)

                    epoch_loss += loss.item()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data), 0))

                progress_bar.update()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data),
                                                                       epoch_span_f1_score))

                self.experiment.log_metric("Span F1", epoch_span_f1_score)

    def __eval_epoch(self, name):
        with torch.no_grad():
            epoch_loss = 0

            self.model.eval()
            with self.experiment.validate():
                with tqdm(total=len(self.eval_data)) as progress_bar:
                    for batch in self.eval_data:
                        tokens = batch[0].to(self.device)
                        tags = batch[1].to(self.device)
                        attention_mask = batch[2].to(self.device)
                        words_ids = batch[3]

                        loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids)

                        epoch_loss += loss.item()

                        progress_bar.update()
                        progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.eval_data), 0))

                    progress_bar.update()
                    epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions)
                    progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.eval_data),
                                                                           epoch_span_f1_score))

    def __test_epoch(self, name: str):
        with torch.no_grad():
            epoch_loss = 0

            self.model.eval()
            with self.experiment.test():
                with tqdm(total=len(self.test_data)) as progress_bar:
                    for batch in self.test_data:
                        tokens = batch[0].to(self.device)
                        tags = batch[1].to(self.device)
                        attention_mask = batch[2].to(self.device)
                        words_ids = batch[3]

                        loss, step_f1 = self.__step(tokens, tags, attention_mask, words_ids)

                        epoch_loss += loss.item()

                        progress_bar.update()
                        progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.test_data), 0))

                    progress_bar.update()
                    epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions)
                    progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.test_data),
                                                                           epoch_span_f1_score))

    def fit(self):
        for epoch in range(self.epochs):
            progress = '[{} / {}] '.format(epoch + 1, self.epochs)

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

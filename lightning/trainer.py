from typing import List, Dict
from ner.utils import clear_for_metrics

import pytorch_lightning as pl
import torch.nn as nn
import torch

from seqeval import f1_score

class LightningBERT(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        # model
        self.model = model

        # make variables for storing true and pred labels from each batch
        self.epoch_labels = []
        self.epoch_predictions = []

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        return optimizer

    def __step(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
               words_ids: List[List[int]]):
        """
        Общий метод, используемый для шага обучения, валидации и тестирования.
        В этом блоке модель делает предсказание по батчу, считается значение функции потерь и полученные предсказания
        сохраняются в списке, чтобы в конце эпохе честно посчитать метрики по сущносят
        """

        predictions = self.model(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))

        predictions = predictions.argmax(dim=2).cpu().numpy()
        labels = labels.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_for_metrics(labels, predictions, self.idx2tag, words_ids)

        self.epoch_labels += clear_labels
        self.epoch_predictions += clear_predictions

        return loss

    def training_step(self, batch, _):
        input_ids, labels, attention_mask, words_ids = batch

        loss = self.__step(input_ids, labels, attention_mask, words_ids)
        metric = f1_score(self.epoch_labels, self.epoch_predictions)

        self.log('Training Batch Step Span F1', metric, prog_bar=True)

        return loss

    def training_epoch_end(self, _):
        epoch_metric = f1_score(self.epoch_true_labels, self.epoch_pred_labels)

        self.log('Training Epoch Span F1', epoch_metric, prog_bar=True)

        self.epoch_labels = []
        self.epoch_predictions = []

    def validation_step(self, batch, _):
        input_ids, labels, attention_mask, words_ids = batch

        loss = self.__step(input_ids, labels, attention_mask, words_ids)
        metric = f1_score(self.epoch_labels, self.epoch_predictions)

        self.log('Validation Batch Step Span F1', metric, prog_bar=True)
        self.log("Validation Loss", loss, prog_bar=True)

        return loss

    def validation_epoch_end(self, _):
        epoch_metric = f1_score(self.epoch_true_labels, self.epoch_pred_labels)

        self.log('Validation Epoch Span F1', epoch_metric, prog_bar=True)

        self.epoch_labels = []
        self.epoch_predictions = []

    def training_step(self, batch, _):
        input_ids, labels, attention_mask, words_ids = batch

        loss = self.__step(input_ids, labels, attention_mask, words_ids)
        metric = f1_score(self.epoch_labels, self.epoch_predictions)

        self.log('Validation Batch Step Span F1', metric, prog_bar=True)
        self.log("Validation Loss", loss, prog_bar=True)

        return loss

    def training_epoch_end(self, _):
        epoch_metric = f1_score(self.epoch_true_labels, self.epoch_pred_labels)

        self.log('Validation Epoch Span F1', epoch_metric, prog_bar=True)

        self.epoch_labels = []
        self.epoch_predictions = []
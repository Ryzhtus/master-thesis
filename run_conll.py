from reader import ReaderCoNLL
from dataset import ChunksDataset
from model import BERT, Baseline, BERTLinear, BERTBase
from trainer import LightningBert
from transformers import BertTokenizer, BertConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from seqeval.scheme import IOBES
import torch


# Список параметров для обучения
params = {
    'model_name': "bert-base-cased",
    'shuffle_train_eval': True,
    'gpus': 1,
    'learning_rate': 1e-5,
    'lstm_lr': 1e-3,
    'adam_eps': 1e-8,
    'lr_scheduler': 'polydecay',
    'maxlen': 256,
    'maxnorm': 1.0,
    'accumulate_grad_batches': 4,
    'max_epochs': 20,
    'weight_decay': 0.02,
    'optimizer': 'torch.adam',
    'warmup_proportion': 0.01,
    'learning_rate_mini': -1,
    'polydecay_ratio': 4, 
    'final_div_factor': 1e4,
    'batch_size': 8,
    'scheme': IOBES,
    'random_seed': 1025,
    "add_features": True,
}

seed_everything(params["random_seed"])

tokenizer = BertTokenizer.from_pretrained(params['model_name'], do_lower_case=False)
device = torch.device('cuda')

reader = ReaderCoNLL()

train_documents = reader.parse("", fit_vectorizer=True)
eval_documents = reader.parse("", fit_vectorizer=True)
test_documents = reader.parse("", fit_vectorizer=True)

train_dataset = ChunksDataset(train_documents, params['maxlen'], tokenizer)
eval_dataset = ChunksDataset(eval_documents, params['maxlen'], tokenizer)
test_dataset = ChunksDataset(test_documents, params['maxlen'], tokenizer)

classes = len(train_dataset.entity_tags)

config = BertConfig()

if params['add_features']:
    model = BERT(config).from_pretrained(params['model_name'])
    logger = WandbLogger(project="NERDocumentFeatures", save_dir=r"")
    logger.log_hyperparams(params)
else:
    model = Baseline(config).from_pretrained(params['model_name'])
    logger = WandbLogger(project="NERDocumentFeatures", save_dir=r"")
    logger.log_hyperparams(params)

lightning_model = LightningBert(model, params, train_dataset.idx2tag, train_dataset, eval_dataset, test_dataset, scheme=params['scheme'])
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

trainer = Trainer(
    gpus=1,
    accelerator='gpu',
    logger=logger,
    accumulate_grad_batches=params['accumulate_grad_batches'],
    max_epochs=params['max_epochs'],
    gradient_clip_val=params['maxnorm'],
    callbacks=[early_stopping],
    val_check_interval=0.25)

trainer.fit(lightning_model)
trainer.test(lightning_model)
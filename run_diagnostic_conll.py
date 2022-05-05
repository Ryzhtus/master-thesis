from pytorch_lightning import Trainer
from reader import ReaderCoNLL
from dataset import ChunksDataset
from trainer import LightningBert
from transformers import BertTokenizer, BertTokenizerFast
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES

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
    'random_seed': 42,
    "add_features": False,
}


tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
tokenizer_fast = BertTokenizerFast.from_pretrained("bert-base-cased")
parser = ReaderCoNLL(tokenizer_fast)
train_documents = parser.parse("data/conll2003/IOB2/train.txt", fit_vectorizer=True)
train_dataset = ChunksDataset(train_documents, params['maxlen'], tokenizer)
diagnostic_documents = parser.parse("data/conll2003/diagnostic/document_context_sentences.txt", is_diagnostic_set=True)
diagnostic_dataset = ChunksDataset(diagnostic_documents, 128, tokenizer)
diagnostic_dataset.idx2tag = train_dataset.idx2tag
diagnostic_dataset.tag2idx = train_dataset.tag2idx
diagnostic_dataloader = DataLoader(diagnostic_dataset, batch_size=8, shuffle=False, collate_fn=diagnostic_dataset.paddings)

pl_model = LightningBert.load_from_checkpoint("logs/NERDocumentFeatures/2edtvbvg/checkpoints/epoch=4-step=2343.ckpt")
pl_model.eval()

logger = WandbLogger(project="NERDocumentFeatures", save_dir=r"logs")

trainer = Trainer(
    logger=logger,
    accelerator='cpu',
    accumulate_grad_batches=params['accumulate_grad_batches'],
    max_epochs=params['max_epochs'],
    gradient_clip_val=params['maxnorm'])

preds = trainer.predict(model=pl_model, dataloaders=diagnostic_dataloader,
                ckpt_path="logs/NERDocumentFeatures/2edtvbvg/checkpoints/epoch=4-step=2343.ckpt")

print(classification_report(pl_model.predict_epoch_labels, pl_model.predict_epoch_predictions, digits=4, mode='strict', scheme=IOBES))
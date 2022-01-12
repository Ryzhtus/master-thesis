from ner.reader import ReaderCoNLL
from ner_lightning.dataset import ChunksDataset

def create_chunk_dataset(filename: str, seq_length: int, tokenizer):
    reader = ReaderCoNLL(include_document_ids=True)
    sentences, labels, _, document2sentences, sentence2position = reader.read(filename)
    dataset = ChunksDataset(sentences, labels, seq_length, document2sentences, sentence2position, tokenizer)
    return dataset
from ner_lightning.utils import create_chunk_dataset

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train = create_chunk_dataset('data/conll2003/IOB2/train.txt', 256, tokenizer)

input_ids, labels, attention_mask, wordpiece_mask = train[2]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

just_value = 15
for input_id, token, label in zip(input_ids, tokens, labels):
    label = label.item()
    converted_label = [train.idx2tag[label]] if label != -100 else tokenizer.pad_token
    print(str(input_id.item()).ljust(just_value), str(token).ljust(just_value), 
          str(label).ljust(just_value), str(converted_label).ljust(just_value))


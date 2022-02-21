from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
print(tokenizer.tokenize("EU"))
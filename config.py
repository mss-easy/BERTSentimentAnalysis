from transformers import BertTokenizer

class SentConfig():
    BERT_PRETRAINED = 'bert-base-uncased'
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PRETRAINED)
    BERT_HIDDEN_SIZE = 768
    MAX_TOKENS_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 32
    EPOCHS = 3
    SEED = 1
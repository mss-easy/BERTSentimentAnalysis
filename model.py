import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import SentConfig

class SentimentModel(nn.Module):
  def __init__(self):
    super(SentimentModel, self).__init__()
    self.bert = BertModel.from_pretrained(SentConfig.BERT_PRETRAINED)
    self.drop_out = nn.Dropout(0.40)  # more dropout value for regularization
    self.linear1 = nn.Linear(SentConfig.BERT_HIDDEN_SIZE, 2)

  def forward(self, input_ids, attention_mask, tt_ids):
    """
    This function takes, BERT Tokenized tweet and passes through
     BERTMOdel layer  -> 40% dropout -> fully connected layer to 2 outputs 
     returns 2 outputs for corresponding sentiment classes
    """
    _, pooled_out = self.bert(input_ids, attention_mask, tt_ids)  
    """ _ = token wise output (ignored), 
        as in our task (sentiment analysis) we need aggregated output """
    out = self.drop_out(pooled_out)
    out = self.linear1(out)
    return out
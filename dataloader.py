from config import SentConfig
import torch

class SentimentDL():
  """
   DataLoader class, it employs mechanishm of tokenisation using bert, truncation and padding .
  :param modified_df: any dataframe train, test, validataion
  :return: DataLoader type object for the provided dataframe 
  """
  def __init__(self, modified_df):
    self.mdf = modified_df

  def __len__(self):
    return self.mdf.shape[0]

  def __getitem__(self, index_num):
    """
    :return: dictionary of the BERT Tokenizer representaion, and the corresponding sentiment represented in long tensor (target varible)
    """
    row = self.mdf.iloc[index_num]
    tweet = row['text']
    # target = [0, 1] if int(row['target']) is 0 else [1, 0] 
    target = 0 if int(row['target']) is 0 else 1
    # {0:'positive class', 1:'negative class'}
    
    tw_bert_tok = SentConfig.TOKENIZER(tweet)

    tw_input_ids = tw_bert_tok['input_ids']
    tw_mask = tw_bert_tok['attention_mask']
    tw_tt_ids = tw_bert_tok['token_type_ids']
    
    len_ = len(tw_input_ids)
    if len_ > SentConfig.MAX_TOKENS_LEN:
      tw_input_ids = tw_input_ids[:SentConfig.MAX_TOKENS_LEN-1]+[102]
      tw_mask = tw_mask[:SentConfig.MAX_TOKENS_LEN]
      tw_tt_ids = tw_tt_ids[:SentConfig.MAX_TOKENS_LEN]
    elif len_ < SentConfig.MAX_TOKENS_LEN:
      pad_len = SentConfig.MAX_TOKENS_LEN - len_
      tw_input_ids = tw_input_ids + ([0] * pad_len)
      tw_mask = tw_mask + ([0] * pad_len)
      tw_tt_ids = tw_tt_ids + ([0] * pad_len)
    
    return {
        'input_ids':torch.tensor(tw_input_ids, dtype=torch.long),
        'attention_mask':torch.tensor(tw_mask, dtype=torch.long),
        'token_type_ids':torch.tensor(tw_tt_ids, dtype=torch.long),
        'target':torch.tensor(target, dtype=torch.long)
    }

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup


from preprocess import preprocess_text
from config import SentConfig
from model import SentimentModel
from dataloader import SentimentDL
from training import train_function, evaluation_function

def preprocess_and_train():
    # read dataset
    data = pd.read_csv('./training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
    data.columns = ('target','uid', 'time', 'query', 'user', 'text')

    # create new dataframe
    sent_df = pd.DataFrame(None, columns=('target', 'text'))
    sent_df['target'] = data['target']
    sent_df['text'] = data['text'].apply(preprocess_text)
    sent_df['tweet_size'] = data['text'].apply(lambda x:len(x.split()))

    # select random sample of 400,000 tweets from total dataset (training on a smaller dataset)
    sent_df_sample = sent_df[(sent_df['tweet_size']>10) & (sent_df['target']==0)].sample(n=200000, random_state=SentConfig.SEED)
    sent_df_sample = sent_df_sample.append(sent_df[(sent_df['tweet_size']>10) & (sent_df['target']==4)].sample(n=200000, random_state=SentConfig.SEED))

    # split dataset into train, test, validation set
    train, test = train_test_split(sent_df_sample, test_size=0.1)
    train, val = train_test_split(train, test_size=0.05)

    # create necessary dataloaders, for advantage of batching by pytorch
    train_dl = SentimentDL(train)
    val_dl = SentimentDL(val)
    test_dl = SentimentDL(test)

    train_loader = DataLoader(train_dl, batch_size=SentConfig.TRAIN_BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_dl, batch_size=SentConfig.VALID_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dl, batch_size=SentConfig.VALID_BATCH_SIZE, shuffle=True)

    # select the cuda device if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create model object
    model = SentimentModel()
    model.to(device)

    # ready with optimizer and scheduler objects 

    # do not apply weight decay in AdamW  to, bias layer and normalization terms
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']  # taken from https://huggingface.co/transformers/training.html 
    # more named parameteres in model.named_parameters()
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optim = AdamW(model.parameters(), lr=5e-5)
    optim = AdamW(optimizer_grouped_parameters, lr=5e-5)

    # learning rate scheduling
    num_train_steps = int((train_dl.__len__()/SentConfig.TRAIN_BATCH_SIZE)*SentConfig.EPOCHS)
    num_warmup_steps = int(0.05*num_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps, num_train_steps)

    # Training : done on the basis of validation loss vs training loss
    
    losses = []
    max_loss = np.inf
    
    for epoch in range(SentConfig.EPOCHS):
        train_loss = train_function(train_loader, model, optim, scheduler, device)
        validation_loss, _ = evaluation_function(validation_loader, model, device)
        losses.append((train_loss, validation_loss))
        print('epoch num: ', epoch, ' train loss:', train_loss, ' validation loss:', validation_loss)
        if validation_loss < max_loss:
                torch.save(model.state_dict(), "SentimentModel.bin")
                max_loss =  validation_loss

    # loss plotting 
    losses = np.array(losses)
    fig, ax = plt.subplots(1, 2, figsize=(14,6))
    ax[0].plot(range(SentConfig.EPOCHS), losses[:,0], 'r')
    ax[1].plot(range(SentConfig.EPOCHS), losses[:,1])
    ax[0].set(xlabel='Epoch num', ylabel='training loss')
    ax[1].set(xlabel='Epoch num', ylabel='validation loss')
    fig.show()

    # F1 score calculation on test predictions

    state_dict_ = torch.load('SentimentModel.bin')
    model = SentimentModel()
    model.load_state_dict(state_dict_)
    model.to(device)
    
    loss_, results = evaluation_function(test_loader, model, device, inference=True)
    print(classification_report(results[:,1], results[:,0]))
    

if __name__ == "__main__":
    preprocess_and_train()
from torch.nn import CrossEntropyLoss
import torch
import numpy as np

loss_function = CrossEntropyLoss()

def train_function(data_loader, model, optimizer, scheduler, device):
  """
  Function to train single epoch on the data accessible with data_loader
  """
  epoch_loss = 0
  model.train()
  for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch['target'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)

        batch_loss = loss_function(outputs, target)
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += batch_loss.item()
  return epoch_loss

def evaluation_function(data_loader, model, device, inference=False):
  """
  Function to evaluate current model performance.
  """
  epoch_loss = 0
  model.eval()

  results = []
  for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch['target'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        if not inference:
          batch_loss = loss_function(outputs, target)
          epoch_loss += batch_loss.item()
        else:
          outputs = torch.argmax(outputs, dim=1).to('cpu').numpy()
          target = target.to('cpu').numpy()
          results.extend(list(zip(outputs, target)))
  return epoch_loss, np.array(results)
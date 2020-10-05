## BERT Sentiment Analysis
A project  on BERT fine tuning for Sentiment Analysis

BERT: It is a tranformer architecture model proposed in: <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>. By this method, we can pre-train general purpose language model and then fine tune it for a given task. BERT takes unlabled text 
as input, mask out 15% words, and tries to predict these words. Masked sentences propogates through transformer architecture. BERT Model also takes care of the relationship between sentences, by training a simple task of predicting, whether the given sentence (in BERT input) is next to previous sentence or not. 

### What I did?

I used pretrained *bert-base-uncased* model and respective BERT Tokenizer. 

Model: I added 2 layers in front of BERT Layer as follows: 40% dropout for regularisation, and then a fully connected linear layer.
Combined architecture:

<img src="image/sentiment_arch_generalised.png">

Above diagram show architecure of my project. It is considered that batch size is 1, though we can input other batch sizes too.

Data Set Used: I used <a href="https://www.kaggle.com/kazanova/sentiment140">sentiment140</a> dataset from kaggle. It contains, 1.6 million tweets. I trained multiple  models on processed subset. I achieved 0.85 F1 score on the random subset of 5L, selected from 1.6M tweets.

I have shared some of the results on kaggle via a notebook:

Kaggle notebook Link: <a href="https://www.kaggle.com/mahendras8894/bert-sentiment-analysis">Bert Sentiment Analysis</a>

### References
1. <a href="https://huggingface.co/transformers/model_doc/bert.html">Transformer Bert Doc</a>
2. <a href="https://huggingface.co/transformers/model_doc/bert.html">Transformer training and fine tuning</a>
3. <a href="https://github.com/google-research/bert">BERT Github</a>
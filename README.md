# Toxic_Comment_Classification
My Trial to tackle the Kaggle Toxic Comment Classification Competition 

I built a model that calculates the probability of a comment belonging to any of the mentioned classes, I used XGBoost after generating feature vectors using GLove and Google news Word2Vec

I got a total AUC of 0.82 

Resources needed:

- Download data from kaggle competition page [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- Download GLove Word Vectors [here](https://nlp.stanford.edu/projects/glove/), choose the 300d.480B model
- Download GoogleNews Word Vectors [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

Note:
```python
final_try.py 
```
file is an implementation to XGBoost algorithm on the same data

To Do::

1. You definetly can make much more hyperparameter optimization epecially regarding the LSTM model.
   for example: You can try playing around with ```max_features```, ```max_len```, ```Droupout_rate```,```size``` of the ```Dense``` layer, etc...
   
2. You can try differnt feature engineering and normaization techniques for the text data
3. In general try playing around with parameters like ```batch_size```, ```num_epochs``` and ```learning_rate``` 
4. Try to use differnt optimization function, maybe ```Adagrad``` ,```Adadelta``` or ```sgd```


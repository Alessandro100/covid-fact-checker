# covid-fact-checker
https://github.com/Alessandro100/covid-fact-checker <br>
Using the Naive Bayes Bag of words model, determine if a given tweet is factual or not

# Running it
Using python 3 in the root directory, run ```python3 main.py```
This will train the model and run it on the test set

# Output
It won't auto generate the .txt files, but will return all the needed metrics in the console

# Naive Bayes vs LSTM
We see that the LSTM is more accurate than the Naive Bayes (filtered / not filtered model). The LSTM model has better precision, recal and F1 scores than any of the Naive Bayes models.

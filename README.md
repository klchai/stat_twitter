# stat_twitter
Project of sentiment analysis on tweets about Apple, Microsoft, Google and Twitter
It creates a file `predictions.txt` which contain the tags predicted by
the ML models for the tweets in the file `test.txt`.
The neural network model is saved in the folder model_nn (if the folder exists, the model
is loaded, else the model is created, fitted and then saved in the folder).

## Prerequisites
Install the libraries in the file `requirements.txt` with the command :
`pip install -r requirements.txt`

## Run the application
`python3 predict_tags.py`

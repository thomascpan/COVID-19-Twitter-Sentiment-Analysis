import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


#Evaluation of CNN model on general Twitter dataset
#Load the preprocessed tweets into a pandas dataframe
preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/CNN/preprocessed.csv"
df = pd.read_csv(preprocessed_tweets, index_col=0)

#For some reason, nulls still exist here so just removed them
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#Split columns into tweets and sentiment labels
x = df.Tweet
y = df.Sentiment_Label

#Split the data into training, testing, validation (80%, 10%, 10%)
SEED = 42
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=.2, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=.5, random_state=SEED)

#Take the top 100000 most common words to limit vocabulary size for faster training
tokenizer = Tokenizer(num_words=100000)
#Create vocabulary index based on word frequency. Lower integers are more frequent words
tokenizer.fit_on_texts(x_train)
#Transform each text in texts to a sequence of integers (from fit_on_texts)
sequences = tokenizer.texts_to_sequences(x_train)
#Make sure all the data is same dimensions because tweets can have different lengths
#Basically add a bunch of 0's so all data is length 45
x_train_seq = pad_sequences(sequences, maxlen=45)

#Same thing that was done on training data, do on validation data
#Transform each text in texts to a sequence of integers (from fit_on_texts)
sequences_val = tokenizer.texts_to_sequences(x_val)
#Make sure all the data is same dimensions because tweets can have different lengths
#Basically add a bunch of 0's so all data is length 45
x_val_seq = pad_sequences(sequences_val, maxlen=45)

#Treat the test data the same way as the training and validation data
#Transform each text in texts to a sequence of integers (from fit_on_texts)
sequences_test = tokenizer.texts_to_sequences(x_test)
#Make sure all the data is same dimensions because tweets can have different lengths
#Basically add a bunch of 0's so all data is length 45
x_test_seq = pad_sequences(sequences_test, maxlen=45)

#Load the best weights out of the epochs of the model just trained
loaded_CNN_model = load_model('CNN_best_weights_sg.hdf5')
#Evaluate the CNN in terms of accuracy
loaded_CNN_model.evaluate(x=x_test_seq, y=y_test, verbose=1)
#Predict the sentiment label so can calculate precision and recall for F1 score
yhat_classes = loaded_CNN_model.predict_classes(x_test_seq)

confusion_matrix = np.array(confusion_matrix(y_test, yhat_classes, labels=[1,0]))
confusion = pd.DataFrame(confusion_matrix, index=["Positive", "Negative"],
                         columns=["Predicted Positive","Predicted Negative"])
print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, yhat_classes)*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(classification_report(y_test, yhat_classes))


#Classify unlabeled COVID-19 tweets based on general Twitter data pretraining
months = ["Jan", "Feb", "Mar", "Apr", "May"]
for month in months:
    #Load the preprocessed tweets into a pandas dataframe
    preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/CNN/%s_preprocessed.csv" % month
    df = pd.read_csv(preprocessed_tweets, index_col=0)

    #For some reason, nulls still exist here so just removed them
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)

    #Split columns into tweets and sentiment labels
    x = df.Tweet
    y = df.Date

    #Take the top 100000 most common words to limit vocabulary size for faster training
    tokenizer = Tokenizer(num_words=100000)
    #Create vocabulary index based on word frequency. Lower integers are more frequent words
    tokenizer.fit_on_texts(x)
    #Transform each text in texts to a sequence of integers (from fit_on_texts)
    sequences = tokenizer.texts_to_sequences(x)
    #Make sure all the data is same dimensions because tweets can have different lengths
    #Basically add a bunch of 0's so all data is length 45
    x_seq = pad_sequences(sequences, maxlen=45)

    #Load the best weights out of the epochs of the model just trained
    #loaded_CNN_model = load_model('CNN_best_weights_sg.hdf5')
    loaded_CNN_model = load_model('CNN_fine_tune3.hdf5')

    #Predict the sentiment label
    yhat_classes = loaded_CNN_model.predict_classes(x_seq)
    sentiment_df = pd.DataFrame(yhat_classes)
    tweet_df = pd.DataFrame(df.Tweet)
    tweet_df["Sentiment"] = sentiment_df
    df_name = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/%s_sentiment_fine.csv" % month
    tweet_df.to_csv(df_name, encoding="utf-8")


#Classify labeled COVID-19 Tweets (small dataset)
#Load the preprocessed tweets into a pandas dataframe
labeled_preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/sentiment_tweets_clean.csv"
df = pd.read_csv(labeled_preprocessed_tweets, index_col=0)

#For some reason, nulls still exist here so just removed them
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#Split columns into tweets and sentiment labels
x = df.Tweet
y = df.Sentiment

#Take the top 100000 most common words to limit vocabulary size for faster training
tokenizer = Tokenizer(num_words=100000)
#Create vocabulary index based on word frequency. Lower integers are more frequent words
tokenizer.fit_on_texts(x)
#Transform each text in texts to a sequence of integers (from fit_on_texts)
sequences = tokenizer.texts_to_sequences(x)
#Make sure all the data is same dimensions because tweets can have different lengths
#Basically add a bunch of 0's so all data is length 45
x_seq = pad_sequences(sequences, maxlen=45)

# Load the best weights out of the epochs of the model just trained
loaded_CNN_model = load_model("CNN_best_weights_sg.hdf5")

# Predict the sentiment label so can calculate precision and recall for F1 score
yhat_classes = loaded_CNN_model.predict_classes(x_seq)

confusion_matrix = np.array(confusion_matrix(y, yhat_classes, labels=[1, 0]))
confusion = pd.DataFrame(confusion_matrix, index=["Positive", "Negative"],
                         columns=["Predicted Positive", "Predicted Negative"])
print("Accuracy: {0:.2f}%".format(accuracy_score(y, yhat_classes) * 100))
print("-" * 80)
print("Confusion Matrix\n")
print(confusion)
print("-" * 80)
print("Classification Report\n")
print(classification_report(y, yhat_classes))



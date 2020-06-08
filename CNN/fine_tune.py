import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#Load the preprocessed tweets into a pandas dataframe
preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/sentiment_tweets_clean.csv"
df = pd.read_csv(preprocessed_tweets, index_col=0)

#For some reason, nulls still exist here so just removed them
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#The dataset is unbalanced
print(df.Sentiment.value_counts()) #411 negative, 308 positive

#Balance out the positive and negative sentiment in the dataset
df_neg = df[df.Sentiment == 0]
df_pos = df[df.Sentiment == 1]
df_neg_sample = df.sample(n=308, replace=False, random_state=1)
df = pd.concat([df_neg_sample, df_pos])

#Split columns into tweets and sentiment labels
x = df.Tweet
y = df.Sentiment

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
model = load_model("CNN_best_weights_sg.hdf5")
#Remove the final classification layer
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
#Fully connected hidden layer = dense
model.add(Dense(256, activation="relu", name="dense_fine2"))
#Models seem to generally add 50% dropout before and after a fully connected hidden layer at the end
#Helps with the generalizability of the models
model.add(Dropout(0.5, name="dropout_fine"))
#Classify into either positive or negative sentiment
model.add(Dense(1, activation="sigmoid", name="dense_fine"))

"""
#If we want to freeze the weights of a few layers
for layer in model.layers[:1]:
    layer.trainable = False
"""

#Compile the CNN and get ready to run it
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#Save the best weights during the 5 epochs of training
filepath = "CNN_fine_tune3.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
model.fit(x_train_seq, y_train, batch_size=8, epochs=10, validation_data=(x_val_seq, y_val), callbacks=[checkpoint])

#Load the best weights out of the epochs of the model just trained
loaded_CNN_model_fine = load_model("CNN_fine_tune3.hdf5")

#Predict the sentiment label so can calculate precision and recall for F1 score
yhat_classes = loaded_CNN_model_fine.predict_classes(x_test_seq)

confusion_matrix_fine = np.array(confusion_matrix(y_test, yhat_classes, labels=[1, 0]))
confusion_fine = pd.DataFrame(confusion_matrix_fine, index=["Positive", "Negative"],
                         columns=["Predicted Positive", "Predicted Negative"])
print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, yhat_classes) * 100))
print("-" * 80)
print("Confusion Matrix\n")
print(confusion_fine)
print("-" * 80)
print("Classification Report\n")
print(classification_report(y_test, yhat_classes))

# Load the best weights out of the epochs of the model just trained
loaded_CNN_model = load_model("CNN_best_weights_sg.hdf5")

#Predict the sentiment label so can calculate precision and recall for F1 score
yhat_classes_general = loaded_CNN_model.predict_classes(x_test_seq)

confusion_matrix = np.array(confusion_matrix(y_test, yhat_classes_general, labels=[1, 0]))
confusion = pd.DataFrame(confusion_matrix, index=["Positive", "Negative"],
                         columns=["Predicted Positive", "Predicted Negative"])
print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, yhat_classes_general) * 100))
print("-" * 80)
print("Confusion Matrix\n")
print(confusion)
print("-" * 80)
print("Classification Report\n")
print(classification_report(y_test, yhat_classes_general))
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score

#Load the pre-trained word embeddings
skip_gram = KeyedVectors.load("w2v_skip_gram.word2vec")

#Load the preprocessed tweets into a pandas dataframe
preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/CNN/preprocessed.csv"
df = pd.read_csv(preprocessed_tweets, index_col=0)

#For some reason, nulls still exist here so just removed them
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#Sample a fraction of the tweets for faster training and testing
#df = df.sample(frac=0.01, replace=False, random_state=1)

#Split columns into tweets and sentiment labels
x = df.Tweet
y = df.Sentiment_Label

#Split the data into training, testing, validation (80%, 10%, 10%)
SEED = 42
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=.2, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=.5, random_state=SEED)

#Make a dictionary of embeddings (useful if try to combine different models)
embeddings_index = {}
for w in skip_gram.wv.vocab.keys():
    embeddings_index[w] = skip_gram.wv[w]

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

#Create the embedding matrix for the embedding layer of the CNN (100000 x 200)
embedding_matrix = np.zeros((100000, 200))
for word, i in tokenizer.word_index.items():
    #Ignore words that are not in the top 100000
    if i >= 100000:
        continue
    #Get the index of the word embedding
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Create a convolutional neural network layer by layer
model = Sequential()
#Create the embedding layer
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)
model.add(e)
#Convolutional layer 1 (Maybe add more with smaller and larger kernels to see if get better results)
#But this configuration got the best results Cliche's SemEval 2017 paper.
model.add(Conv1D(filters=200, kernel_size=3, padding="valid", activation="relu", strides=1))
#Convolutional layer 2
model.add(Conv1D(filters=200, kernel_size=4, padding="valid", activation="relu", strides=1))
#Convolutional layer 3
model.add(Conv1D(filters=200, kernel_size=5, padding="valid", activation="relu", strides=1))
#Use global max pooling to get only the most important features (more compact, speeds up)
#GlobalMaxPooling2D is wrong dimension...don't keep making that mistake
model.add(GlobalMaxPooling1D())
#Models seem to generally add 50% dropout before and after a fully connected hidden layer at the end
model.add(Dropout(0.5))
#Fully connected hidden layer = dense
model.add(Dense(256, activation="relu"))
#Models seem to generally add 50% dropout before and after a fully connected hidden layer at the end
#Helps with the generalizability of the models
model.add(Dropout(0.5))
#Classify into either positive or negative sentiment
model.add(Dense(1, activation="sigmoid"))

#Compile the CNN and get ready to run it
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#Save the best weights during the 5 epochs of training
filepath = "CNN_best_weights_sg.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
model.fit(x_train_seq, y_train, batch_size=32, epochs=5, validation_data=(x_val_seq, y_val), callbacks=[checkpoint])

#Load the best weights out of the epochs of the model just trained
loaded_CNN_model = load_model('CNN_best_weights_sg.hdf5')

#Treat the test data the same way as the training and validation data
#Transform each text in texts to a sequence of integers (from fit_on_texts)
sequences_test = tokenizer.texts_to_sequences(x_test)
#Make sure all the data is same dimensions because tweets can have different lengths
#Basically add a bunch of 0's so all data is length 45
x_test_seq = pad_sequences(sequences_test, maxlen=45)

#Evaluate the CNN in terms of accuracy
loaded_CNN_model.evaluate(x=x_test_seq, y=y_test, verbose=1)
#Predict the sentiment label so can calculate precision and recall for F1 score
yhat_classes = loaded_CNN_model.predict_classes(x_test_seq)
print("F1 Score: %f" % f1_score(y_test, yhat_classes))
print("Accuracy: %f" % accuracy_score(y_test, yhat_classes))
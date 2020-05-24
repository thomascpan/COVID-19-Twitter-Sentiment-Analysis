import pandas as pd
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils

#Load the preprocessed tweets into a pandas dataframe
preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/CNN/preprocessed.csv"
df = pd.read_csv(preprocessed_tweets, index_col=0)

#For some reason, nulls still exist here so just removed them
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#Split columns into tweets and sentiment labels
x = df.Tweet
y = df.Sentiment_Label

def label_tweets(tweets, group_label):
    """
    :param tweets: a pandas data frame column of tweets
    :param group_label: a label to tag the tweet (along with unique row id)
    :return: a TaggedDocument iterable of tweet words associated with a unique row id
    """
    word2vec_input = []
    #Iterate through all the tweets and associate each with its row number as its id
    for i, tweet in enumerate(tweets):
        try:
            #Split the tweet into words
            tweet_words = tweet.split()
            #Create a TaggedDocument iterable of the form:
            #(words = list of words in the tweet, tags = tag_row number of tweet)
            #The tag is a string token associated with a particular tweet
            tagged_tweets = TaggedDocument(tweet_words, [group_label + "_%s" % i])
            #Add the labeled tweet to the list
            word2vec_input.append(tagged_tweets)
        except:
            continue
    return word2vec_input

#Label the tweets
all_x_w2v = label_tweets(x, "corpus")

#Choose skip-gram over continuous bag of words to deal with infrequent words better

#Use the Word2Vec API to create 200 dimension word embeddings of the corpus
#sg = 1 stands for training algorithm for skip gram, 0 = continous bag of words
#size = 200 means the word embeddings have 200 dimensions
#negative = 5 means how many noise words should be drawn when negative sampling. Documentation: between 5 and 20
#window = 2 means maximum distance between the current and predicted word within a sentence
#min_count = 2 means to ignore all words with total frequency lower than this threshold
#alpha = 0.065 means the initial learning rate
#min_alpha = 0.05 means the learning rate will drop to here linearly as training progresses
skip_gram = Word2Vec(sg=1, size=200, negative=5, window=5, min_count=2, alpha=0.065, min_alpha=0.065)

#Build a vocabulary from a sequence of sentences
#Just give it a list words from the TaggedDocuments iterable
skip_gram.build_vocab([x.words for x in tqdm(all_x_w2v)])

#Train for 30 epochs
#Not sure why but setting epochs in train to 30 instead of using the for loop causes everything to be really slow
#after the second epoch so just did this instead
for epoch in range(30):
    skip_gram.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    skip_gram.alpha -= 0.002
    skip_gram.min_alpha = skip_gram.alpha

#Save the word embeddings
skip_gram.save("w2v_skip_gram.word2vec")


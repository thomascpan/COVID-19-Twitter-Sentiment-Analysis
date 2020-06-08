import pandas as pd

months = ["Jan", "Feb", "Mar", "Apr", "May"]
labels = [0, 1]
samples = []
for month in months:
    cols = ["Num", "Date", "Tweet", "Sentiment"]
    #Load the preprocessed tweets into a pandas dataframe
    preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/%s_tweets_labeled.csv" % month
    df = pd.read_csv(preprocessed_tweets, header=None, names=cols)

    #Only text and sentiment needed
    df.drop(["Num", "Date"], axis=1, inplace=True)
    df.drop([0], axis=0, inplace=True)

    #Remove nulls that stop further preprocessing
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    samples.append(df)

random_df = pd.concat(samples)
#Write out the cleaned dataset
random_df.to_csv("C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/sentiment_tweets.csv")

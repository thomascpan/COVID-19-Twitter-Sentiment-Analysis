import pandas as pd

#Read in the training datafile
cols = ["Date", "Tweet"]
train_data = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/PracticeData/filtered_final.csv"
df = pd.read_csv(train_data, header=0, names=cols, encoding="latin-1")

#Remove nulls that stop further preprocessing
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

#Extract the tweets that have a month in their date
months = ["Jan", "Feb", "Mar", "Apr", "May"]
df_list = []
#Separate out the tweets months
for month in months:
    tweet_date_df = df[df["Date"].str.contains(month)]
    df_list.append(tweet_date_df)

#Create list to store thetweets
tweets = []
month_tweets = []

#Iterate through each month's dataset, add each tweet to the list
for i in range(len(df_list)):
    for j in range(df_list[i].shape[0]):
        #Add it to the clean tweet list
        tweets.append(df_list[i].iloc[j, 1])
        month_tweets.append(months[i])

#Store the clean tweets in a pandas dataframe with their sentiment classification
tweet_df = pd.DataFrame(tweets, columns=["Tweet"])
months_df = pd.DataFrame(month_tweets, columns=["Date"])
months_df["Tweet"] = tweet_df

#Remove an null rows that result. These are rows that only had twitter ID or url address
tweet_df.dropna(inplace=True)
tweet_df.reset_index(drop=True, inplace=True)

for month in months:
    dir = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/"
    df_name = dir + "%s_tweets.csv" % month
    months_df[months_df["Date"].isin([month])].to_csv(df_name, encoding="utf-8")
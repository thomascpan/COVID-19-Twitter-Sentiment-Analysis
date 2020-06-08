import pandas as pd
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May"]
ratios = []
for month in months:
    #Load the preprocessed tweets into a pandas dataframe
    preprocessed_tweets = "C:/Users/travi/OneDrive/Documents/UCLA/CS263/Final Project/Sentiment/%s_sentiment_fine.csv" % month
    df = pd.read_csv(preprocessed_tweets, index_col=0)

    #Get the total number of tweets
    num_tweets = df.shape[0]
    #Get the sentiment column
    num_pos = sum(df["Sentiment"])

    ratio = num_pos / num_tweets
    ratios.append(ratio)

plt.plot(months, ratios)
plt.xlabel("Months")
plt.ylabel("Percent Positive Sentiment")
plt.show()

df = pd.DataFrame(months, columns=["Month"])
df_pos = pd.DataFrame(ratios)
df["Positive Percent"] = df_pos
df.to_csv("Final_CNN_fine.csv", encoding="utf-8")
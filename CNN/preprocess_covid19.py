import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer

#Tokenize a tweet into a sequence of alphabetic and non-alphabetic characters via regexp \w+|[^\w\s]+
tok = WordPunctTokenizer()

#Deal with the @mention by taking only alphanumerics
at_mention = r"@[A-Za-z0-9_]+"
#Remove URL links
url = r"https?://[^ ]+"
at_url = r"|".join((at_mention, url))
www = r"www.[^ ]+"
#A lot of tweets seem to begin with rt, remove with regex
rt = r"^rt "

#WordPunctTokenizer will remove apostrophes and the t after it
#Thus negative contractions like "can't" will become can, which has reverse sentiment
#This holds conversions
negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                 "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                 "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                 "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                 "mustn't": "must not"}

#Make a regex pattern to search for these negation contractions (keys) and replace them with the
#full length negation (values) in the negation dictionary
#\b = word boundary
#A|B where A and B are arbitrary regex expression, a regular expression will match either A or B
neg_pattern = re.compile(r"\b(" + "|".join(negations_dic.keys()) + r")\b")

def preprocess_tweet(tweet):
    """
    :param tweet: A row from the dataset representing a tweet to preprocess (string)
    :return: The cleaned tweet with HTML, @, #, URL, negation abbreviations, removed and tokenized
    """
    #Convert HTML encoding into text using lxml as an HTML parser
    #lxml has external dependency so needed to install separately, but its fast
    soup = BeautifulSoup(tweet, "lxml")
    #Get only the human-readable text inside a document or tag
    souped = soup.get_text()
    #Avoid getting strange character patterns like \xef\xbf\xbd in your tweets
    try:
        #Remove UTF-8 Byte Order Marks (used so reader can identify a file as UTF-8)
        no_html = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        no_html = souped
    #Replace @mention and URLs with nothing
    stripped = re.sub(at_url, "", no_html)
    stripped = re.sub(www, "", stripped)
    #Lower case all the letters (no-op for non alphabetic)
    lower_case = stripped.lower()
    #Use regex to remove all the negative contractions
    #group returns the string that occurs after regex substitution
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    #Remove the "#" from hashtags
    letters_only = re.sub("[^A-Za-z]", " ", neg_handled)
    #Tokenize the tweets and save in list comprehension
    #tokenize returns a tokenized copy of the string s (avoid nulls by checking length)
    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    #Join the tokens together separated by spaces, remove whitespace to complete cleaned tweet
    final_tokens = (" ".join(words)).strip()
    #Remove the random rt in the beginning of a lot of tweets
    final_tokens_no_rt = re.sub(rt, "", final_tokens)
    return final_tokens_no_rt

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

#Create list to store the preprocessed tweets
clean_tweets = []
month_clean_tweets = []

#Iterate through each month's dataset, preprocess each tweet and add it to the list
for i in range(len(df_list)):
    for j in range(df_list[i].shape[0]):
        #Preprocess the tweet
        #The "Tweet" column is index 1
        preprocessed = preprocess_tweet(df_list[i].iloc[j, 1])
        #Add it to the clean tweet list
        clean_tweets.append(preprocessed)
        month_clean_tweets.append(months[i])

#Store the clean tweets in a pandas dataframe with their sentiment classification
clean_df = pd.DataFrame(clean_tweets, columns=["Tweet"])
months_df = pd.DataFrame(month_clean_tweets, columns=["Date"])
clean_df["Date"] = months_df

print(clean_df.head())

#Remove an null rows that result. These are rows that only had twitter ID or url address
clean_df.dropna(inplace=True)
clean_df.reset_index(drop=True, inplace=True)

for month in months:
    df_name = "%s_preprocessed.csv" % month
    clean_df[clean_df["Date"].isin([month])].to_csv(df_name, encoding="utf-8")
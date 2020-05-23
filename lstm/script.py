import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import csv
from sentiment_lstm import SentimentLSTM
from tqdm import tqdm


def preprocess(df: pd.core.frame.DataFrame) -> None:
    """ Preprocess dataset.
        - text
            - lowercase text
            - remove punctuations
        - sentiment
            - remove neutral (1-3) values
            - map to 0(neg) and 1(pos)
    Args:
        df (pd.core.frame.DataFrame): dataframe object
    """
    # lowercase
    df.text = df.text.str.lower()
    # remove punctuation
    df.text = df.text.str.replace('[{}]'.format(punctuation), '')
    # remove neutral (temp)
    df.drop(df.loc[~df.sentiment.isin([0, 4])].index, inplace=True)
    # map labels (temp)
    df.sentiment = df.sentiment.map({0: 0, 4: 1})


def plot(tweets_int: list) -> None:
    """ Plots distribution of tweet lengths and outputs some descriptive statistics. 
        - text
            - lowercase text
            - remove punctuations
        - sentiment
            - remove neutral (1-3) values
            - map to 0(neg) and 1(pos)
    Args:
        tweets_int (list): list consisting of the length of each tweet
    """
    tweets_len = [len(x) for x in tweets_int]
    pd.Series(tweets_len).hist()
    plt.show()
    print(pd.Series(tweets_len).describe())


def pad_features(tweets_int: list, seq_length: int) -> np.ndarray:
    """ Returns features of tweet_ints, where each tweet is padded with 0's or truncated to the input seq_length.
    Args:
        tweets_int (list): list consisting of the length of each tweet
        seq_length (int): length of features.
    Returns:
        np.ndarray: matrix of features padded with zeroes at the front.
    """
    features = np.zeros((len(tweets_int), seq_length), dtype=int)
    for i, tweet in enumerate(tweets_int):
        tweet_len = len(tweet)
        if tweet_len <= seq_length:
            zeroes = list(np.zeros(seq_length - tweet_len))
            new = zeroes + tweet
        elif tweet_len > seq_length:
            new = tweet[0:seq_length]
        features[i, :] = np.array(new)
    return features


def preprocess_tweet(tweet: str) -> str:
    """ Preprocess tweet
        - text
            - lowercase text
            - remove punctuations
    Args:
        tweet (str): tweet to be evaluated
    Returns:
        str: preprocessed tweet
    """
    return ''.join([c for c in tweet.lower() if c not in punctuation])


def tokenize_tweets(tweets: list, vocab_to_int: dict) -> list:
    """ Tokenize tweets with vocab_to_int.
    Args:
        tweet (str): tweet to be evaluated
        vocab_to_int (dict): dict of vocab mapped to their order based on word count
    Returns:
        list: a list tokenized tweets
    """
    return [tokenize_tweet(tweet, vocab_to_int) for tweet in tweets]


def tokenize_tweet(tweet: str, vocab_to_int: dict) -> list:
    """ Tokenize tweet with vocab_to_int.
    Args:
        tweet (str): tweet to be evaluated
        vocab_to_int (dict): dict of vocab mapped to their order based on word count
    Returns:
        list: a tokenized tweet
    """
    return [vocab_to_int[word] for word in tweet.split()]


def predict(net, vocab_to_int, tweet, sequence_length=200):
    """ Predict sentiment of tweet based on model
    Args:
        net ():
        vocab_to_int (int): length of features.
        tweet (str): length of features.
        sequence_length (int): length of features.
    Returns:
        np.ndarray: matrix of features padded with zeroes at the front.
    """
    net.eval()

    # Preprocess tweet
    tweet = preprocess_tweet(tweet)

    # Tokenize tweet
    tweets_int = tokenize_tweets([tweet], vocab_to_int)

    # Create padded features.
    features = pad_features(tweets_int, sequence_length)

    # Convert to tensor
    feature_tensor = torch.from_numpy(features)

    # Initialize hidden state
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output negative or positive (0 or 1)
    pred = torch.round(output.squeeze())

    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if(pred.item() == 1):
        print("Positive tweet detected!")
    else:
        print("Negative tweet detected.")


def save_model(filepath, model) -> None:
    torch.save(model.state_dict(), filepath)


def load_model(filepath, model) -> None:
    model.load_state_dict(torch.load(filepath))


def main():
    # Load Data and preprocess
    headers = ["sentiment", "text"]
    df = pd.read_csv('data/data.csv', header=None,
                     names=headers, usecols=[0, 5], encoding='latin-1')
    preprocess(df)

    # Tokenize: Vocab to int mapping ordered based on count
    sorted_words = Counter(df.text.str.split(
        expand=True).stack().value_counts().to_dict()).most_common()
    vocab_to_int = {w: i for i, (w, c) in enumerate(sorted_words)}

    # Extract tweets and labels
    tweets = df.text.to_list()
    labels = df.sentiment.to_numpy()

    # maps each word in a tweet to its vocab_to_int mapping.
    tweets_int = tokenize_tweets(tweets, vocab_to_int)

    # plot(tweets_int)

    # Create padded features.
    seq_length = 50
    features = pad_features(tweets_int, seq_length)

    # Create training, validation, and test dataset.
    len_feat = len(features)
    split_frac = 0.8

    train_x = features[0:int(split_frac * len_feat)]
    train_y = labels[0:int(split_frac * len_feat)]

    remaining_x = features[int(split_frac * len_feat):]
    remaining_y = labels[int(split_frac * len_feat):]

    valid_x = remaining_x[0:int(len(remaining_x) * 0.5)]
    valid_y = remaining_y[0:int(len(remaining_y) * 0.5)]

    test_x = remaining_x[int(len(remaining_x) * 0.5):]
    test_y = remaining_y[int(len(remaining_y) * 0.5):]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(
        train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(
        valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(
        test_x), torch.from_numpy(test_y))

    # Loads, shuffles, and batches data.
    batch_size = 32
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True,
                             batch_size=batch_size, drop_last=True)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
    output_size = 1
    embedding_dim = 32
    hidden_dim = 64
    n_layers = 2
    net = SentimentLSTM(vocab_size, output_size,
                        embedding_dim, hidden_dim, n_layers)

    # loss and optimization functions
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # training params
    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing
    clip = 5  # gradient clipping

    net.train()

    # train for some number of epochs
    outer = tqdm(total=epochs, desc="Epoch", position=0)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        train_inner = tqdm(total=len(train_loader.dataset),
                           desc="Batch: Train", position=0)
        for inputs, labels in train_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            net.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            inputs = inputs.type(torch.LongTensor)

            # Step 3. Run our forward pass.
            output, h = net(inputs, h)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            train_inner.update()

        # Get validation loss
        val_h = net.init_hidden(batch_size)
        val_losses = []
        net.eval()

        valid_inner = tqdm(total=len(valid_loader.dataset),
                           desc="Batch: Validation", position=0)
        for inputs, labels in valid_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])

            inputs = inputs.type(torch.LongTensor)
            output, val_h = net(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

            valid_inner.update()

        net.train()
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Loss: {:.6f}...".format(loss.item()),
              "Val Loss: {:.6f}".format(np.mean(val_losses)))

        outer.update()

    # Get test data loss and accuracy
    test_losses = []  # track loss
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    test_inner = tqdm(total=len(test_loader.dataset),
                      desc="Batch: Test", position=0)
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get predicted outputs
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

        test_inner.update()

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    model_path = "model/model.pt"
    save_model(net)
    print("Saving model")

    tweet = 'This movie had the best acting and the dialogue was so good. I loved it.'
    predict(net, vocab_to_int, tweet, seq_length)


if __name__ == "__main__":
    main()

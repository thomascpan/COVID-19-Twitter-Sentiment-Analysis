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


def sample_features_and_labels(features: np.ndarray, labels: np.ndarray, percent: int) -> (np.ndarray, np.ndarray):
    """ Randomly samples from features.
    Args:
        features (np.ndarray): list consisting of the length of each tweet
        percent (float): percent of features to sample (decimal)
    Returns:
        np.ndarray: sampled features
    """
    n = features.shape[0]
    size = int(n * percent)
    rand_indices = np.random.choice(n, size=size, replace=False)
    return (features[rand_indices, :], labels[rand_indices])


def sample_predictions(df: pd.core.frame.DataFrame, n: int) -> pd.core.frame.DataFrame:
    """ Randomly sample from predictions.
    Args:
        df (pd.core.frame.DataFrame): df of predictions
        n (float): number of samples
    Returns:
        np.ndarray: sampled features
    """
    return df.sample(n)


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
    return [vocab_to_int.get(word, 0) for word in tweet.split()]


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

    return int(pred.item())


def save_model(filepath, model) -> None:
    torch.save(model.state_dict(), filepath)


def load_model(filepath, model) -> None:
    model.load_state_dict(torch.load(filepath))


def precision(tp, fp) -> float:
    return tp / (tp + fp)


def recall(tp, fn) -> float:
    return tp / (tp + fn)


def f1(precision, recall) -> float:
    return 2 * (precision * recall) / (precision + recall)


def predict_task(net, vocab_to_int, seq_length, filepath):
    tqdm.pandas()
    read_columns = ["created_at", "text", "lang"]
    write_columns = ["created_at", "text"]

    df = pd.read_csv(filepath, usecols=read_columns)
    mask = ((df.text.str.len() <= 140) & (df.lang == "en"))
    df = df.loc[mask]
    df.created_at = pd.to_datetime(df['created_at'])
    df.index = pd.to_datetime(df['created_at'], format='%m/%d/%y %I:%M%p')
    sampled_df = sample_predictions(df, 10)
    df["sentiment"] = df.progress_apply(lambda row: predict(
        net, vocab_to_int, row["text"], seq_length), axis=1)
    sampled_df["sentiment"] = sampled_df.progress_apply(lambda row: predict(
        net, vocab_to_int, row["text"], seq_length), axis=1)
    print(df.groupby([df.index.month, df.sentiment]).agg({'count'}))
    sampled_df.to_csv("results/sampled_results.csv", index=False)
    print(sampled_df)


def main():
    # Load Data and preprocess
    headers = ["sentiment", "text"]
    df = pd.read_csv('data/data.csv', header=None,
                     names=headers, usecols=[0, 5], encoding='latin-1')
    preprocess(df)

    # Tokenize: Vocab to int mapping ordered based on count
    sorted_words = Counter(df.text.str.split(
        expand=True).stack().value_counts().to_dict()).most_common()
    vocab_to_int = {w: i for i, (w, c) in enumerate(sorted_words, 1)}

    # Extract tweets and labels
    tweets = df.text.to_list()
    tweet_labels = df.sentiment.to_numpy()

    # maps each word in a tweet to its vocab_to_int mapping.
    tweets_int = tokenize_tweets(tweets, vocab_to_int)

    # plot(tweets_int)

    # Create padded features.
    seq_length = 50
    features = pad_features(tweets_int, seq_length)

    sampled_features, sampled_labels = sample_features_and_labels(
        features, tweet_labels, 0.10)

    # Create training, validation, and test dataset.
    len_feat = len(sampled_features)
    split_frac = 0.8

    train_x = sampled_features[0:int(split_frac * len_feat)]
    train_y = sampled_labels[0:int(split_frac * len_feat)]

    remaining_x = sampled_features[int(split_frac * len_feat):]
    remaining_y = sampled_labels[int(split_frac * len_feat):]

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
    epoch_pbar = tqdm(total=epochs, desc="Epoch", position=0)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        train_pbar = tqdm(total=len(train_loader.dataset),
                          desc="Batch: Train", position=1)
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

            train_pbar.update(batch_size)

        # Get validation loss
        val_h = net.init_hidden(batch_size)
        val_losses = []
        net.eval()

        validation_pbar = tqdm(total=len(valid_loader.dataset),
                               desc="Batch: Validation", position=2)
        for inputs, labels in valid_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])

            inputs = inputs.type(torch.LongTensor)
            output, val_h = net(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float())

            val_losses.append(val_loss.item())

            validation_pbar.update(batch_size)

        net.train()

        epoch_pbar.write("Epoch: {}/{}...".format(e + 1, epochs))
        epoch_pbar.write("Loss: {:.6f}...".format(loss.item()))
        epoch_pbar.write("Val Loss: {:.6f}".format(np.mean(val_losses)))

        epoch_pbar.update()

    epoch_pbar.close()
    train_pbar.close()
    validation_pbar.close()

    # Get test data loss and accuracy
    test_losses = []  # track loss
    num_correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    test_pbar = tqdm(total=len(test_loader.dataset),
                     desc="Batch: Test", position=4)
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
        label_tensor = labels.float().view_as(pred)
        correct_tensor = pred.eq(label_tensor)
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

        tp += ((correct_tensor == 1) & (label_tensor == 1)).numpy().sum()
        fp += ((correct_tensor == 1) & (label_tensor == 0)).numpy().sum()
        tn += ((correct_tensor == 0) & (label_tensor == 0)).numpy().sum()
        fn += ((correct_tensor == 0) & (label_tensor == 1)).numpy().sum()

        test_pbar.update(batch_size)

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    precision_score = precision(tp, fp)
    recall_score = recall(tp, fn)
    f1_score = f1(precision_score, recall_score)
    model_path = "model/model.pt"
    save_model(model_path, net)

    # avg test loss
    test_pbar.write("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_pbar.write("Test accuracy: {:.3f}".format(test_acc))
    test_pbar.write("Precision: {:.3f}".format(precision_score))
    test_pbar.write("Recall: {:.3f}".format(recall_score))
    test_pbar.write("F1: {:.3f}".format(f1_score))
    test_pbar.write("Saving model")

    test_pbar.close()

    predict_task(net, vocab_to_int, seq_length, "data/final.csv")


if __name__ == "__main__":
    main()

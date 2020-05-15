import config
import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import csv

from sentiment_model_define_class import SentimentLSTM


def preprocess(df):
    # lowercase
    df[2] = df[2].str.lower()
    # remove punctuation
    df[2] = df[2].str.replace('[{}]'.format(punctuation), '')
    # remove neutral (temp)
    df.drop(df.loc[df[1] == "neutral"].index, inplace=True)
    # map labels (temp)
    # df[1] = df[1].map({"neutral": 0, "positive": 1, "negative": -1})
    df[1] = df[1].map({"positive": 1, "negative": 0})


def plot(reviews_int: list) -> None:
    reviews_len = [len(x) for x in reviews_int]
    pd.Series(reviews_len).hist()
    plt.show()
    print(pd.Series(reviews_len).describe())


def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]
        features[i, :] = np.array(new)
    return features


def tokenize_review(test_review, vocab_to_int):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints


def predict(net, vocab_to_int, test_review, sequence_length=200):

    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review, vocab_to_int)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if(config.train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if(pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")


def longest_text(x):
    mx = 0
    idx = 0
    for i, v in enumerate(x):
        l = len(v.split())
        if l > mx:
            mx = l
            idx = i
    return x[idx]


def main():
    df = pd.read_csv('data/data.txt',
                     sep='\t', header=None, usecols=[0, 1, 2], quoting=csv.QUOTE_NONE)
    preprocess(df)

    reviews = df[2].to_list()
    labels = df[1].to_numpy()

    sorted_words = Counter(df[2].str.split(
        expand=True).stack().value_counts().to_dict()).most_common()

    vocab_to_int = {w: i for i, (w, c) in enumerate(sorted_words)}

    reviews_int = []
    for review in reviews:
        reviews_int.append([vocab_to_int[w] for w in review.split()])

    # plot(reviews_int)

    seq_length = 40

    features = pad_features(reviews_int, seq_length)

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

    # dataloaders
    batch_size = 50
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True,
                              batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True,
                             batch_size=batch_size, drop_last=True)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
    output_size = 1
    embedding_dim = 50
    hidden_dim = 256
    n_layers = 2
    net = SentimentLSTM(vocab_size, output_size,
                        embedding_dim, hidden_dim, n_layers)

    # loss and optimization functions
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    # move model to GPU, if available
    if(config.train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(config.train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(config.train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.LongTensor)
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    # Get test data loss and accuracy

    test_losses = []  # track loss
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        if(config.train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

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
        correct = np.squeeze(correct_tensor.numpy(
        )) if not config.train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    test_review = 'This movie had the best acting and the dialogue was so good. I loved it.'
    predict(net, vocab_to_int, test_review, seq_length)


if __name__ == "__main__":
    main()

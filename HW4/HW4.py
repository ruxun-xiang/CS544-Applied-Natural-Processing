import os
import pandas as pd
import numpy as np
from torch.utils import data
import torch.nn.functional as F
from torch import nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.autograd import Variable

# path = "/content/drive/MyDrive/CS544/HW4"
# os.chdir(path)

# create vocabulary and label projection
train_path = "./data/train"
f = open(train_path)

vocab = []
labels = []
vocab_dict = {}
labels_dict = {}

occurance = {}
for i in f:
    if str(i) != "\n":
        index, word, label = i.split(" ")
        label = label.split("\n")[0]

        occurance[word] = occurance.get(word, 0) + 1

        if word not in vocab_dict.keys():
            vocab_dict[word] = len(vocab_dict)

        if label not in labels_dict.keys():
            labels_dict[label] = len(labels_dict)

# Add PAD and UNK index
vocab_dict["PAD"] = len(vocab_dict)
vocab_dict["UNK"] = len(vocab_dict)

reverse_vocab = {}
for key, value in vocab_dict.items():
    reverse_vocab[value] = key

reverse_label = {}
for key, value in labels_dict.items():
    reverse_label[value] = key
print("create vocabulary and label projection ok")

# create train file corpus (representing in index)

train_path = "./data/train"
f = open(train_path)

sentence = []
sentence_label = []

corpus = []
corpus_label = []

word_count = 0
for i in f:
    if str(i) != "\n":
        word_count += 1
        index, word, label = i.split(" ")
        label = label.split("\n")[0]
        if occurance[word] < 2:
            sentence.append(vocab_dict["UNK"])
        else:
            sentence.append(vocab_dict[word])
        sentence_label.append(labels_dict[label])
    else:
        corpus.append(sentence)
        corpus_label.append(sentence_label)
        sentence = []
        sentence_label = []

corpus.append(sentence)
corpus_label.append(sentence_label)

# create dev set corpus
dev_path = "./data/dev"
f = open(dev_path)

dev_sent = []
dev_label = []

dev_corpus = []
dev_corpus_label = []
max_word = 0
word_count = 0
count = 0
for i in f:
    if str(i) != "\n":
        index, word, label = i.split(" ")
        word_count += 1
        label = label.split("\n")[0]
        if word in vocab_dict:
            dev_sent.append(vocab_dict[word])
        else:
            dev_sent.append(vocab_dict["UNK"])
        dev_label.append(labels_dict[label])
        count += 1
    else:
        count += len(dev_sent)
        if int(index) > max_word:
            max_word = int(index)

        dev_corpus.append(dev_sent)
        dev_corpus_label.append(dev_label)
        dev_sent = []
        dev_label = []

dev_corpus.append(dev_sent)
dev_corpus_label.append(dev_label)

# create test set corpus
test_path = "./data/test"
f = open(test_path)

test_sent = []
test_label = []
test_corpus = []
test_corpus_label = []
max_word = 0
word_count = 0
count = 0
for i in f:
    if str(i) != "\n":
        index, word = i.split(" ")
        word = word.split("\n")[0]
        label = "O"
        if word in vocab_dict:
            test_sent.append(vocab_dict[word])
        else:
            test_sent.append(vocab_dict["UNK"])
        test_label.append(labels_dict[label])
    else:
        if int(index) > max_word:
            max_word = int(index)
        test_corpus.append(test_sent)
        test_corpus_label.append(test_label)
        test_sent = []
        test_label = []

test_corpus.append(test_sent)
test_corpus_label.append(test_label)

print("create corpus ok")

# create X and y from train set
max_len = 113

corpus_data_pad = vocab_dict['PAD'] * np.ones((len(corpus), max_len))
corpus_label_pad = -1 * np.ones((len(corpus), max_len))

for j in range(len(corpus)):
    cur_len = len(corpus[j])
    corpus_data_pad[j][:cur_len] = corpus[j]
    corpus_label_pad[j][:cur_len] = corpus_label[j]

corpus_data_pad, corpus_label_pad = torch.LongTensor(corpus_data_pad), torch.LongTensor(corpus_label_pad)

corpus_data_pad, corpus_label_pad = Variable(corpus_data_pad), Variable(corpus_label_pad)

# create X and y from dev set
max_len = 109

dev_corpus_data_pad = vocab_dict['PAD'] * np.ones((len(dev_corpus), max_len))
dev_corpus_label_pad = -1 * np.ones((len(dev_corpus), max_len))

for j in range(len(dev_corpus)):
    cur_len = len(dev_corpus[j])
    dev_corpus_data_pad[j][:cur_len] = dev_corpus[j]
    dev_corpus_label_pad[j][:cur_len] = dev_corpus_label[j]
    # print(corpus_label_pad[j])

dev_corpus_data_pad, dev_corpus_label_pad = torch.LongTensor(dev_corpus_data_pad), torch.LongTensor(
    dev_corpus_label_pad)
dev_corpus_data_pad, dev_corpus_label_pad = Variable(dev_corpus_data_pad), Variable(dev_corpus_label_pad)

# create X from test set

max_len = 124
test_corpus_data_pad = vocab_dict['PAD'] * np.ones((len(test_corpus), max_len))
test_corpus_label_pad = -1 * np.ones((len(test_corpus), max_len))
for j in range(len(test_corpus)):
    cur_len = len(test_corpus[j])
    test_corpus_data_pad[j][:cur_len] = test_corpus[j]
    test_corpus_label_pad[j][:cur_len] = test_corpus_label[j]

test_corpus_data_pad = torch.LongTensor(test_corpus_data_pad)
test_corpus_label_pad = torch.LongTensor(test_corpus_label_pad)
test_corpus_data_pad = Variable(test_corpus_data_pad)
test_corpus_label_pad = Variable(test_corpus_label_pad)

print("create X and y ok")


# used for dataloader
class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.label = y
        self.data = X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]

        return X, y


# Blstm model 1
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # embedding input (batch, sent_len)
        # need padding
        # embedding output (batch, sent_len, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Linear(output_dim, 9)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax()

    def forward(self, X):
        # embedding
        embeds = self.embedding(X)
        embeds = self.dropout(embeds)

        # lstm layer
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[2])

        # linear layer
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)

        # Classifier
        fc_out = self.fc(elu_out)
        softmax = F.log_softmax(fc_out, dim=1)

        return softmax


# test dev set loss
def dev_loss():
    dev_loss = 0.0
    with torch.no_grad():
        for data, label in test_generator:
            data = Variable(data).cuda()
            labels = Variable(label).cuda()

            labels = labels.view(-1)
            outputs = lstm(data)

            crit = nn.NLLLoss(ignore_index=-1)
            loss = crit(outputs, labels)
            dev_loss += loss.item() * data.size(0)
        dev_loss = dev_loss / len(test_generator.dataset)
        print('Dev Loss: {:.6f}'.format(dev_loss), end=" ")


# test dev set accuracy
def report_acc():
    total = 0
    correct = 0
    prediction = None
    with torch.no_grad():
        for data, label in dev_generator:
            data = Variable(data).cuda()
            labels = Variable(label).cuda()

            labels = labels.view(-1)

            mask = (labels >= 0).float()
            num_tokens = int(torch.sum(mask).item())

            a = labels.cpu().numpy()
            non_padding_index = np.where(a >= 0)
            none_padding_label = a[np.array(non_padding_index).T,]
            none_padding_label = none_padding_label.reshape(none_padding_label.shape[0])

            outputs = lstm(data)  # dim: batch_size*batch_max_len x lstm_hidden_dim

            b = outputs.cpu().numpy()

            non_padding_output = b[np.array(non_padding_index).T,]
            non_padding_output = non_padding_output.reshape(non_padding_output.shape[0], non_padding_output.shape[2])
            mi = np.argmax(non_padding_output, axis=1)
            maxi = mi.reshape(-1, 1)

            if prediction is None:
                prediction = maxi
            else:
                prediction = np.vstack((prediction, maxi))

            total += len(none_padding_label)
            correct += np.sum(mi == none_padding_label)
            acc = correct / total
        print('Accuracy on the test set: %.6s' % (correct / total))
    return prediction, acc


def prediction_on_test():
    prediction = None
    with torch.no_grad():
        for data, label in test_generator:
            data = Variable(data).cuda()
            labels = Variable(label).cuda()

            labels = labels.view(-1)

            mask = (labels >= 0).float()
            num_tokens = int(torch.sum(mask).item())

            a = labels.cpu().numpy()
            non_padding_index = np.where(a >= 0)
            none_padding_label = a[np.array(non_padding_index).T,]
            none_padding_label = none_padding_label.reshape(none_padding_label.shape[0])

            outputs = lstm(data)  # dim: batch_size*batch_max_len x lstm_hidden_dim

            b = outputs.cpu().numpy()

            non_padding_output = b[np.array(non_padding_index).T,]
            non_padding_output = non_padding_output.reshape(non_padding_output.shape[0], non_padding_output.shape[2])
            mi = np.argmax(non_padding_output, axis=1)
            maxi = mi.reshape(-1, 1)

            if prediction is None:
                prediction = maxi
            else:
                prediction = np.vstack((prediction, maxi))
    return prediction, acc


# predict on dev set and output file
def predict_over_dev(prediction, epoc, task):
    prediction = prediction.tolist()
    print("record prediction")
    predict_file = "./data/dev" + task + ".out"
    dev_file = "./data/dev"
    fdev = open(dev_file)
    fout = open(predict_file, "w")
    j = 0
    for i in fdev:
        if str(i) != "\n":
            index, word, label = i.split(" ")
            label = label.split("\n")[0]
            label_pred = prediction[j]
            # print(label_pred)
            line = index + " " + word + " " + label + " " + reverse_label[label_pred[0]]
            j += 1
            fout.write(line)
            fout.write("\n")
        else:
            fout.write("\n")


# predict on test set and output file
def predict_over_test(prediction, task):
    prediction = prediction.tolist()
    predict_file = "./data/test" + task + ".out"
    test_file = "./data/test"
    ftest = open(test_file)
    fout = open(predict_file, "w")
    j = 0
    for i in ftest:
        if str(i) != "\n":
            index, word = i.split(" ")
            word = word.split("\n")[0]
            label_pred = prediction[j]
            line = index + " " + word + " " + reverse_label[label_pred[0]]
            j += 1
            fout.write(line)
            fout.write("\n")
        else:
            fout.write("\n")


# Blstm1 prediction

# Hyper param
batch_size = 192
vocab_size = len(vocab_dict)
num_layers = 1
embedding_dim = 100
hidden_dim = 256
output_dim = 128
dropout = 0.33
lr = 0.8
epoch = 1000

X_train = corpus_data_pad
y_train = corpus_label_pad

X_dev = dev_corpus_data_pad
y_dev = dev_corpus_label_pad

X_test = test_corpus_data_pad
y_test = test_corpus_label_pad

train_set = Dataset(X_train, y_train)

train_generator = data.DataLoader(train_set, batch_size=batch_size)

dev_set = Dataset(X_dev, y_dev)
dev_generator = data.DataLoader(dev_set, batch_size=batch_size)

test_set = Dataset(X_test, y_test)
test_generator = data.DataLoader(test_set, batch_size=batch_size)

lstm = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, dropout, num_layers).cuda()

criterion = nn.NLLLoss(ignore_index=-1)
optimizer = torch.optim.SGD(lstm.parameters(), lr=lr, momentum=0.9)

blstm1_model_path = "./model/blstm1.pt"
# load model
checkpoint = torch.load(blstm1_model_path)
lstm.load_state_dict(checkpoint["net"])
lstm.eval()

dev_prediction, acc = report_acc()  # record dev prediction
test_prediction, acc = prediction_on_test()  # record test prediction
predict_over_dev(dev_prediction, 0, "1")  # output prediction file
predict_over_test(test_prediction, "1")  # output prediction file

print("predict using blstm1 ok")


# Blstm2 modeling
class LSTMModel_glove(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, num_layers):
        super(LSTMModel_glove, self).__init__()
        self.hidden_dim = hidden_dim

        # embedding input (batch, sent_len)
        # need padding
        # embedding output (batch, sent_len, embedding_dim)
        #        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embed))
        # LSTM模块使用word_embeddings作为输入，输出的维度为hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)

        # nn.Linear将LSTM模块的输出映射到目标向量空间，即线性空间
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Linear(output_dim, 9)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax()

    def forward(self, X):
        embeds = self.embedding(X)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[2])

        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)

        # dropout = self.dropout(fc_out)

        # Classifier
        fc_out = self.fc(elu_out)
        softmax = F.log_softmax(fc_out, dim=1)

        return softmax


# glove embedding
glove_path = "./data/glove.6B.100d"
words_embed = {}
f = open(glove_path)

vocab_large = vocab_dict.copy()

for line in f:
    aline = line.split()
    word = aline[0]
    if word not in vocab_large.keys():
        idx = len(vocab_large)
        vocab_large[word] = idx

    embed = aline[1:]
    embed = [float(num) for num in embed]
    words_embed[word] = embed

reverse_vocab_large = {}

for key, value in vocab_large.items():
    reverse_vocab_large[value] = key

id2emb = {}
count = 0
for ix in range(len(vocab_large)):
    word = reverse_vocab_large[ix]

    if word != "PAD" and word != "UNK":
        if word in words_embed:
            id2emb[ix] = words_embed[reverse_vocab_large[ix]]
            count += 1
        elif word.lower() in words_embed:
            id2emb[ix] = words_embed[word.lower()]
            count += 1
        else:
            id2emb[ix] = [0.0] * 100
    else:
        id2emb[ix] = [0.0] * 100
glove_embed = [id2emb[ix] for ix in range(len(vocab_large))]

batch_size = 192
vocab_size = len(vocab_large)
num_layers = 1
embedding_dim = 100
hidden_dim = 256
output_dim = 128
dropout = 0.33
lr = 0.5
epoch = 1000

X_train = corpus_data_pad
y_train = corpus_label_pad

X_dev = dev_corpus_data_pad
y_dev = dev_corpus_label_pad

X_test = test_corpus_data_pad
y_test = test_corpus_label_pad

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = Dataset(X_train, y_train)

train_generator = data.DataLoader(train_set, batch_size=batch_size)

dev_set = Dataset(X_dev, y_dev)
dev_generator = data.DataLoader(dev_set, batch_size=batch_size)

test_set = Dataset(X_test, y_test)
test_generator = data.DataLoader(test_set, batch_size=batch_size)

lstm = LSTMModel_glove(vocab_size, embedding_dim, hidden_dim, output_dim, dropout, num_layers).cuda()

# criterion = Loss_fn()
criterion = nn.NLLLoss(ignore_index=-1)
optimizer = torch.optim.SGD(lstm.parameters(), lr=lr, momentum=0.9)

blstm2_model_path = "./model/blstm2.pt"
checkpoint = torch.load(blstm2_model_path)
lstm.load_state_dict(checkpoint["net"])
lstm.eval()
dev_prediction, acc = report_acc()
test_prediction, acc = prediction_on_test()  # record test prediction

predict_over_dev(dev_prediction, 0, "2")
predict_over_test(test_prediction, "2")

print("predict using blstm2 ok")




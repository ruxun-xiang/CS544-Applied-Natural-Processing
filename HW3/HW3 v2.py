#!/usr/bin/env python
# coding: utf-8

# In[53]:


import json
from tqdm import tqdm


# # Task 1: Vocabulary Creation

# In[54]:


# record occurrence of every word
data_path = "./data/train"
f = open(data_path, "r")

word_dic = {}
unk = "<unk>"
unknum = "<unkn>"
unkword = "<unkw>"
tag_list = []
for line in f:
    if line != "\n":
        index, word, tag = line.split("\t")
        word_dic[word] = word_dic.get(word, 0) + 1

# create a new train set with word occurred < 2 replaced by <unk> tag
data_path = "./data/train"
f = open(data_path, "r")
output_path = "./data/train_rep_unk"
out = open(output_path, "w")

for line in f:
    if line != "\n":
        index, word, tag = line.split("\t")
        if word_dic[word] < 2:
            # classify the unknown words
            if "0" < word < "9":
                out.write(index + "\t" + unknum + "\t" + tag) 
            else:
                out.write(index + "\t" + unkword + "\t" + tag)
        out.write(index + "\t" + word + "\t" + tag)
    elif line == "\n" or line == "":
        out.write("\n")

# use the new train and calculate word occurrence again(including unk)
data_path = "./data/train_rep_unk"
f = open(data_path)

word_dic_unk = {}
for line in f:
    if line != "\n":
        index, word, tag = line.split("\t")
        tag = tag.split("\n")[0]
        word_dic_unk[word] = word_dic_unk.get(word, 0) + 1
        if tag not in tag_list:
            tag_list.append(tag)
f.close()

print("Total size of my vocabulary:{}".format(len(word_dic_unk)))
print("Total occurrences of the special token <unk>:{}".format(word_dic_unk[unknum] + word_dic_unk[unkword]))

# reorder the words based on their occurrence
word_dic_order = sorted(word_dic_unk.items(), key=lambda x: x[1], reverse=True)

# create vocab.txt using the ordered vocab
index = 1
output_file = "./data/vocab.txt"
f = open(output_file, "w")
f.write(unk + "\t" + str(index) + "\t" + str(word_dic_unk[unkword] + word_dic_unk[unknum]))
f.write("\n")
index += 1
for word, occur in word_dic_order:
    if word != unknum and word != unkword:
        f.write(word + "\t" + str(index) + "\t" + str(occur))
        f.write("\n")
        index += 1

f.close()


# # Task 2: Model Learning

# In[55]:


# create hmm.json
train_file = "./data/train_rep_unk"
f = open(train_file)

cnt_s1s2 = {}
cnt_sx = {}

cnt_s = {}

start_tag = "<s>"

# compute transition and emission
def compute_ts(sentence):
    sent_len = len(sentence)
    for i in range(sent_len):
        if sent_len == 1:
            first_i, first_x, first_s = sentence[i].split("\n")[0].split("\t")
            cnt_s1s2[(start_tag, first_s)] = cnt_s1s2.get((start_tag, first_s), 0) + 1
            cnt_sx[(first_s, first_x)] = cnt_sx.get((first_s, first_x), 0) + 1
            cnt_s[first_s] = cnt_s.get(first_s, 0) + 1
            cnt_s[start_tag] = cnt_s.get(start_tag, 0) + 1

        if i + 1 < sent_len:
            j = i + 1
            first_i, first_x, first_s = sentence[i].split("\n")[0].split("\t")
            sec_i, sec_x, sec_s = sentence[j].split("\n")[0].split("\t")

            cnt_s1s2[(first_s, sec_s)] = cnt_s1s2.get((first_s, sec_s), 0) + 1
            cnt_sx[(first_s, first_x)] = cnt_sx.get((first_s, first_x), 0) + 1
            cnt_s[first_s] = cnt_s.get(first_s, 0) + 1
            # if it's first word, append a key (<s>, first_word)
            if i == 0:
                cnt_s1s2[(start_tag, first_s)] = cnt_s1s2.get((start_tag, first_s), 0) + 1
                cnt_s[start_tag] = cnt_s.get(start_tag, 0) + 1
            if j == sent_len - 1:
                cnt_s[sec_s] = cnt_s.get(sec_s, 0) + 1
                cnt_sx[(sec_s, sec_x)] = cnt_sx.get((sec_s, sec_x), 0) + 1

sentence = []
for line in f:
    if line != "\n":
        sentence.append(line)
    elif line == "\n" or line == "":
        compute_ts(sentence)
        sentence = []
f.close()

# record string key for storation in json file
# use original key for later computation
transition_str = {}
transition = {}
emission_str = {}
emission = {}

ts = {}
alpha = 0.001
for key, value in cnt_s1s2.items():
    s1 = key[0]
    s2 = key[1]
    s1s2 = "(" + s1 + "," + s2 + ")"
    transition_str[s1s2] = (value + alpha) / ( cnt_s[key[0]] + alpha * len(cnt_s1s2))
    # smoothing
    transition[key] = (value + alpha) / ( cnt_s[key[0]] + alpha * len(cnt_s1s2))

for key, value in cnt_sx.items():
    s = key[0]
    x = key[1]
    sx = "(" + s + "," + x + ")"
    emission_str[sx] = (value + alpha) / ( cnt_s[key[0]] + alpha * len(cnt_sx))
    # smoothing
    emission[key] = (value + alpha) / ( cnt_s[key[0]] + alpha * len(cnt_sx))

print("Total number of transition parameters:{}".format(len(transition)))
print("Total number of emission parameters:{}".format(len(emission)))

hmm = {}
hmm['transition'] = transition_str
hmm['emission'] = emission_str
hmm_file = "./data/hmm.json"
fout = open(hmm_file, "w")
fout.write(json.dumps(hmm))
fout.close()


# # Task 3: Greedy Decoding with HMM

# ## Greedy for evaluation

# In[56]:


def greedy_acc(sentence):
    s = []
    s_t = []
    curr_tag = ""
    correct = 0
    sent_len = len(sentence)
    for i in range(sent_len):
        maxp = -1
        index, word, wtag = sentence[i].split("\t")
        wtag = wtag.split("\n")[0]
        if word not in word_dic_unk.keys():
            if "0" < word < "9":
                word = unknum
            else:
                word = unkword

        # if first word, s_prime should be <s>
        if i == 0:
            s_prime = start_tag
        else:
            s_prime = s[i - 1]

        pred_tag = ""
        for tag in tag_list:
            t_val = 0
            e_val = 0
            e_key = (tag, word)
            t_key = (s_prime, tag)
            # for all possible situations, first find suitable t
            if t_key in transition.keys():
                t_val = transition[t_key]
                # then find suitable e
                if e_key in emission.keys():
                    e_val = emission[e_key]
            p = t_val * e_val
            # record max p, record the corresponding predicted tag
            if p > maxp:
                maxp = p
                pred_tag = tag
        s.append(pred_tag)
        # compute correct prediction
        if pred_tag == wtag:
            correct += 1

    return correct


# ## Evaluate greedy on dev data

# In[57]:


dev_file = "./data/dev"
f = open(dev_file)
sentence = []
corrects = 0
length = 0
counter = 0
dev_count = 0
for line in f:
    dev_count += 1
    if line != "\n":
        sentence.append(line)

    elif line == "\n" or line == "":
        length += len(sentence)
        correct = greedy_acc(sentence)
        corrects += correct
        sentence = []
        counter += 1

print("Greedy decoding accuracy on dev data:{:.1f}%".format(corrects * 100 / length))


# ## Greedy for prediction

# In[58]:


import pandas as pd

def greedy_pos(sentence, fout):
    s = []
    s_t = []
    curr_tag = ""
    correct = 0
    sent_len = len(sentence)
    for i in range(sent_len):
        maxp = -1
        index, word = sentence[i].split("\t")
        word = word.split("\n")[0]
        ori_word = word
        if word not in word_dic_unk.keys():
            if "0" < word < "9":
                word = unknum
            else:
                word = unkword

        if i == 0:
            s_prime = start_tag
        else:
            s_prime = s[i - 1]

        pred_tag = ""
        for tag in tag_list:
            t_val = 0
            e_val = 0
            e_key = (tag, word)
            t_key = (s_prime, tag)

            if t_key in transition.keys():
                t_val = transition[t_key]
                if e_key in emission.keys():
                    e_val = emission[e_key]
            p = t_val * e_val
            if p > maxp:
                maxp = p
                pred_tag = tag
        s.append(pred_tag)
        # write prediction in file
        fout.write(index + "\t" + ori_word + "\t" + pred_tag)
        fout.write("\n")


    return 0


# ## Predict POS in test data

# In[59]:


test_file = "./data/test"
f = open(test_file)
output_file = "./data/greedy.out"
fout = open(output_file, "w")
sentence = []
corrects = 0
length = 0
counter = 0
test_count = 0
for line in f:
    test_count += 1
    if line != "\n":
        sentence.append(line)
    elif line == "\n" or line == "":
        flag = greedy_pos(sentence, fout)
        fout.write("\n")
        sentence = []

f.close()
fout.close()


# # Task 4: Viterbi Decoding with HMM

# ## Viterbi for evaluation

# In[60]:


def viterbi_acc(sentence):
    pi = {}
    path = {}
    best = ""
    sent_len = len(sentence)
    wtags = []
    correct = 0

    for i in range(sent_len):
        pi_val = {}
        pi_path = {}
        index, word, wtag = sentence[i].split("\t")
        wtag = wtag.split("\n")[0]
        wtags.append(wtag)

        if word not in word_dic_unk.keys():
            if "0" < word < "9":
                word = unknum
            else:
                word = unkword

        if i == 0:
            s_prime = start_tag
            for tag in tag_list:
                t_val = 0
                e_val = 0
                e_key = (tag, word)
                t_key = (s_prime, tag)
                # first find suitable t
                if t_key in transition.keys():
                    t_val = transition[t_key]
                    # then find suitable e
                    if e_key in emission.keys():
                        e_val = emission[e_key]
                # record pi value and path that led to this value
                pi_val[tag] = t_val * e_val
                pi_path[tag] = [tag]
            # record the first layer pi value and path
            pi[i] = pi_val
            path[i] = pi_path

        else:
            for curr_tag in tag_list:
                pi_val[curr_tag] = -1

                for prev_tag in tag_list:
                    prev_pi = pi[i - 1][prev_tag]

                    curr_pi = 0

                    if prev_pi != 0:
                        t_val = 0
                        e_val = 0
                        e_key = (curr_tag, word)
                        t_key = (prev_tag, curr_tag)

                        if t_key in transition.keys():
                            t_val = transition[t_key]
                            if e_key in emission.keys():
                                e_val = emission[e_key]
                        # record current pi for later comparison
                        curr_pi = prev_pi * t_val * e_val
                    # if for the tag in this layer, current pi value greater than max in record, 
                    # replace and record the current pi value and the path that led to this value
                    if curr_pi > pi_val[curr_tag]:
                        pi_val[curr_tag] = curr_pi
                        pi_path[curr_tag] = path[i - 1][prev_tag][:]
                        pi_path[curr_tag].append(curr_tag)
            # record the pi values and paths for this layer
            pi[i] = pi_val
            path[i] = pi_path
    # find the best pi value in the last layer, and the corresponding path
    best_pi = max(pi[sent_len - 1], key=pi[sent_len - 1].get)
    best_path = path[sent_len - 1][best_pi]
    # compute correct prediction
    for i in range(len(best_path)):
        if best_path[i] == wtags[i]:
            correct += 1

    return correct


# ## Evaluate Viterbi on dev data

# In[61]:


f = open(dev_file)
sentence = []
corrects = 0
length = 0
counter = 0
for line in tqdm(f, total=dev_count):
    if line != "\n":
        sentence.append(line)
    elif line == "\n" or line == "":
        length += len(sentence)
        correct = viterbi_acc(sentence)
        corrects += correct
        sentence = []

print("Viterbi decoding accuracy on dev data:{:.1f}%".format(corrects * 100 / length))


# ## Viterbi for prediction

# In[62]:


def viterbi_pos(sentence, fout):
    pi = {}
    path = {}
    best = ""
    sent_len = len(sentence)
    words = []
    correct = 0


    for i in range(sent_len):
        pi_val = {}
        pi_path = {}
        index, word = sentence[i].split("\t")
        word = word.split("\n")[0]
        words.append(word)

        if word not in word_dic_unk.keys():
            if "0" < word < "9":
                word = unknum
            else:
                word = unkword

        if i == 0:
            s_prime = start_tag
            for tag in tag_list:
                t_val = 0
                e_val = 0
                e_key = (tag, word)
                t_key = (s_prime, tag)
                if t_key in transition.keys():
                    t_val = transition[t_key]
                    if e_key in emission.keys():
                        e_val = emission[e_key]
                pi_val[tag] = t_val * e_val
                pi_path[tag] = [tag]
            pi[i] = pi_val
            path[i] = pi_path

        else:
            for curr_tag in tag_list:
                pi_val[curr_tag] = -1

                for prev_tag in tag_list:
                    prev_pi = pi[i - 1][prev_tag]

                    curr_pi = 0

                    if prev_pi != 0:
                        t_val = 0
                        e_val = 0
                        e_key = (curr_tag, word)
                        t_key = (prev_tag, curr_tag)

                        if t_key in transition.keys():
                            t_val = transition[t_key]
                            if e_key in emission.keys():
                                e_val = emission[e_key]

                        curr_pi = prev_pi * t_val * e_val
                    if curr_pi > pi_val[curr_tag]:
                        pi_val[curr_tag] = curr_pi
                        pi_path[curr_tag] = path[i - 1][prev_tag][:]
                        pi_path[curr_tag].append(curr_tag)

            pi[i] = pi_val
            path[i] = pi_path
    best_pi = max(pi[sent_len - 1], key=pi[sent_len - 1].get)

    best_path = path[sent_len - 1][best_pi]
    # write prediction into file
    for i in range(len(best_path)):
        fout.write(str(i + 1) + "\t" + words[i] + "\t" + best_path[i])
        fout.write("\n")

    return 0


# ## Predict POS in test data

# In[63]:


test_file = "./data/test"
f = open(test_file)
output_file = "./data/viterbi.out"
fout = open(output_file, "w")
sentence = []
corrects = 0
length = 0
counter = 0
for line in tqdm(f, total=test_count):
    if line != "\n":
        sentence.append(line)
    elif line == "\n" or line == "":
        flag = viterbi_pos(sentence, fout)
        fout.write("\n")
        sentence = []

f.close()
fout.close()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import pandas as pd
import csv

'''
2.1 General design
- Read input and randomly initialize word embeddings
- Generate input (with context window w=0,1)
- neural network training (hidden_layer=1, layer_width=128, nonlinearity=tanh)

2.2 Feature Engineering
- add features to input vectors
- Read orig files

2.3 Pretrained Embeddings
- Read embedding from twitter
'''

# The code below referenced the pytorch tutorial 
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html


def read_text_input(train_dev_devtest, w): 
    '''
    Read from text file and generate a list of input to feed to neural network.
    The input is of the form [[context word, center word, context word], tag]
    '''

    vocab = set()
    tags = set()
    processed_input = []
    current_sentence = ["<s>"] * w
    current_sentence_tags = []

    with open("./twpos-data/twpos-"+train_dev_devtest+".tsv", 'r') as f: 
        while True: 
            line = f.readline()
            if not line: 
                break

            if len(line) != 1: # not empty line
                word, tag = line.strip().split('\t')
                vocab.add(word)
                tags.add(tag)
                current_sentence.append(word)
                current_sentence_tags.append(tag)

            else: 
                current_sentence += ["<\\s>"] * w
                l = len(current_sentence) - w * 2
                processed_input += [[current_sentence[i:i+2*w+1], current_sentence_tags[i]] for i in range(l)]
                current_sentence = ["<s>"] * w
                current_sentence_tags = []

    vocab.add("<s>")
    vocab.add("<\\s>")
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    vocab_size = len(word_to_ix)
    word_to_ix = defaultdict(lambda:vocab_size, word_to_ix)

    return word_to_ix, processed_input


tags = [
    'N', 'O', 'S', 'L', '^', 'Z', 'M', 'V', 'A', 'R', '!',
    'D', 'P', '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E',
    '$', ',', 'G'
]
tags_to_ix = {tag: i for i, tag in enumerate(tags)}



def read_embeddings(filename="./twitter-embeddings.txt"): 
    '''
    Read an existing set of embeddings. Output new word-to-index
    dictionary and the tensor that contains all embeddings
    '''

    df = pd.read_table(
        filename, 
        sep=" ", 
        delimiter=' ', 
        header=None, 
        quoting=3, 
        keep_default_na=False)

    ix_to_words = df[0].to_dict()
    pretrained_words_to_ix = {v: k for k, v in ix_to_words.items()}

    # trained the tag <s> from scratch
    pretrained_words_to_ix['<s>'] = len(pretrained_words_to_ix)
    pretrained_vocab_size = len(pretrained_words_to_ix)


    # words that we have not seen would be mapped to the embedding at index pretrained_vocab_size
    pretrained_words_to_ix = defaultdict(
        lambda:pretrained_vocab_size, pretrained_words_to_ix)
    
    weight = torch.tensor(df.iloc[:, 1:].values)

    # randomly initialize weights for unseen words and <s>
    weight = torch.cat(
        (weight, torch.FloatTensor(2, 50).uniform_(-0.01, 0.01)))

    return pretrained_words_to_ix, weight


pretrained_words_to_ix, pretrained_weights = read_embeddings()


def compute_features(word): 
    '''
    Add the following feature vectors
    - (binary) whether the center word contains ed
    - (binary) whether the center word contains ing
    - (binary) whether the center word contains ly
    - (binary) whether the center word contains any special characters
    - (binary) whether the center word contains any digits
    - (count) number of characters in the center word

    Input: 
        word: (str) the center word
    
    Output: (tensor) feature vector
    '''

    result = torch.tensor(["ed" in word, "ing" in word, "ly" in word, any(
        not c.isalnum() for c in word), any(
        c.isdigit() for c in word), len(word)/10], dtype=torch.float)
    return torch.reshape(result, (1, 6))


class POSTagger(nn.Module): 

    def __init__(
        self, 
        vocab_size, 
        num_tags, 
        embedding_dim, 
        w, 
        num_layers, 
        layer_width, 
        pretrained, 
        add_features): 

        super(POSTagger, self).__init__()
        if not pretrained: 
            self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
            self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else: 
            self.embeddings = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False)

        if num_layers == 0: 
            self.linears = nn.ModuleList([nn.Linear(
                (w * 2 + 1) * embedding_dim +add_features*6, num_tags)])
        else: 
            self.linears = nn.ModuleList([nn.Linear((
                w * 2 + 1) * embedding_dim + add_features*6, layer_width)])
            for _ in range(num_layers-1): 
                self.linears.append(nn.Linear(layer_width, layer_width))
            self.linears.append(nn.Linear(layer_width, num_tags))


    def forward(self, inputs, add_features, word): 
        '''
        Input: 
            inputs: (tensor) word indices
            add_features: (bool) True if we include features to the input vectors
            word: (string) the center word

        Output: (tensor) log probabilities for all outcomes
        '''

        embeds = self.embeddings(inputs).view((1, -1))
        if add_features: 
            # print(embeds.shape, compute_features(word).shape)
            embeds = torch.cat((embeds, compute_features(word)), 1)

        if len(self.linears) == 1: 
            out = self.linears[0](embeds.float())
        else: 
            out = F.relu(self.linears[0](embeds.float()))
            for lin in self.linears[1:-1]: 
                out = F.relu(lin(out))
            out = self.linears[-1](out)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs



def train_nn(
    w, 
    embedding_dim=50,
    num_layers=1, 
    layer_width=128, 
    lr=0.02, 
    epoch_num=10, 
    pretrained=False, 
    add_features=False): 
    '''
    Train word embeddings using neural network

    Input: 
        w: (int) window size 
        embedding_dim: (int) dimension of embeddings
        num_layers: (int) number of layers
        layer_width: (int) width of layer
        lr: (float) learning rate
        epoch_num: (int) number of epochs
        pretrained: (bool) True if using pretrained embeddings
    '''

    # Read input
    notrained_word_to_ix, train_processed_input = read_text_input(
        "train", w)
    _, dev_processed_input = read_text_input(
        "dev", w)
    _, devtest_processed_input = read_text_input(
        "devtest", w)
    vocab_size = len(notrained_word_to_ix)
    num_tags = len(tags_to_ix)
    

    # initialize loss and establish the model and optimizer
    losses = []
    loss_function = nn.NLLLoss()
    max_dev_accuracy = 0
    model = POSTagger(
        vocab_size, 
        num_tags, 
        embedding_dim, 
        w, 
        num_layers, 
        layer_width, 
        pretrained, 
        add_features)
    optimizer = optim.SGD(model.parameters(), lr)

    if pretrained: 
        word_to_ix = pretrained_words_to_ix
    else: 
        word_to_ix = notrained_word_to_ix


    # start the SGD process
    for _ in range(epoch_num): 
        total_loss = 0
        for x, y in train_processed_input: 
            idxs = torch.tensor([word_to_ix[wd] for wd in x], dtype=torch.long)
            model.zero_grad()
            center_word = x[len(x) // 2]
            log_probs = model(idxs, add_features, center_word)
            loss = loss_function(log_probs, torch.tensor([
                tags_to_ix[y]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # report loss and dev accuracy for each epoch
        print(total_loss)
        dev_accuracy = calculate_accuracy(
            dev_processed_input, model, word_to_ix, add_features)
        if dev_accuracy > max_dev_accuracy: 
            max_dev_accuracy = dev_accuracy
            devtest_accuracy = calculate_accuracy(devtest_processed_input, model, word_to_ix, add_features)
            print("Best dev accuracy is {}, the corresponding devtest accuracy is {}".format(dev_accuracy, devtest_accuracy))
        losses.append(total_loss)
    print(losses)
    print(devtest_accuracy)

    # calculate_accuracy(dev_processed_input, model, word_to_ix, add_features, gather_inaccuracy=True)

    return max_dev_accuracy, devtest_accuracy



def calculate_accuracy(devinput, model, word_to_ix, add_features, gather_inaccuracy=False): 
    '''
    Input: 
        devinput: (list) dev input generated by read_text_input
        model: torch model
        word_to_ix: (dictionary) word to index for the given model
        add_features: (bool) whether the model includes added features or not
        gather_inaccuracy: (bool) whether to store the mistakes file to csv or not

    Output: (float) accuracy percentage
    '''


    # referenced this post: https://discuss.pytorch.org/t/how-to-find-test-accuracy-after-training/88962/3
    num_correct = 0
    num_samples = 0

    mistakes = []

    for x, y in devinput: 
        idxs = torch.tensor([word_to_ix[wd] for wd in x], dtype=torch.long)
        center_word = x[len(x) // 2]
        scores = model(idxs, add_features, center_word)
        _, predictions = scores.max(1)
        num_correct += (predictions == tags_to_ix[y]).sum()
        num_samples += 1
        if predictions != tags_to_ix[y]: 
            mistakes.append([center_word, y])


    # save mistakes to csv file for feature design inspirations
    if gather_inaccuracy: 
        with open("mistakes.csv", 'w') as f: 
            write = csv.writer(f)
            write.writerows(mistakes)

    return num_correct / num_samples



'''
for num_layers in [0,1,2]: 
    for layer_width in [256, 512]: 
        print("Number of layers: {}. Layer width: {}".format(num_layers, layer_width))
        train_nn(1, num_layers=num_layers, layer_width=layer_width, pretrained=True, add_features=True)

train_nn(2)
'''





import pandas as pd
import re
from nltk import trigrams, bigrams
from collections import defaultdict
import random
import pickle
import numpy as np

small_file = "small_set.csv"
dev_file = "dev_set.csv"
test_file = "test_set.csv"
train_file = "train_set.csv"

#***************** GLOBAL VARIABLES *****************
# Vocab num for testing set (used in add-one)
train_vocab_count = 0

# the language models trained
add_one_model = defaultdict(lambda: defaultdict(lambda:0))
si_model = defaultdict(lambda: defaultdict(lambda:0))

# n-gram counts in the form of a model
trigram_model = defaultdict(lambda: defaultdict(lambda: 1))
bigram_model = defaultdict(lambda: defaultdict(lambda: 1))
unigram_model = defaultdict(lambda: 1)
#****************************************************

def main():
    """
    :return: Prints out information I need - perplexity
    """
    global trigram_model, bigram_model, unigram_model, train_vocab_count

    #---------------------- Begin work on training data------------------------
    # train_string = extract_text(train_file)
    # train_sentences, less_than_five, train_vocab_count = training_work(train_string)
    # pickle.dump((train_sentences, less_than_five, train_vocab_count), open("train_sentences.pkl", 'wb'))

    ##################################################################### TODO: Only comment out if already ran code before!!!
    train_sentences, less_than_five, train_vocab_count = pickle.load(open("train_sentences.pkl", 'rb'))
    #####################################################################
    print("Training sentences extracted from file")
    print("Training Vocab Count is: " + str(train_vocab_count))# Value = 6121

    #----------------- Build trigram, bigram, and unigram counts --------------
    for s in train_sentences:
        for w1, w2, w3 in trigrams(s.split()):
            trigram_model[(w1, w2)][w3] += 1
        for w1, w2 in bigrams(s.split()):
            bigram_model[w1][w2] += 1
        for w1 in s.split():
            unigram_model[w1] += 1
    print("N-gram calculations complete")

    # ---------------------- Begin work on testing data------------------------
    # test_string = extract_text(dev_file) # TODO: This can be used if trying to use dev file. Otherwise, default to bottom
    # test_sentences = testing_work(test_string, less_than_five)
    # pickle.dump(test_sentences, open("test_sentences.pkl", 'wb'))

    # test_string = extract_text(test_file) # TODO: Default to loading this as the test set
    # test_sentences = testing_work(test_string, less_than_five)
    # pickle.dump(test_sentences, open("real_test_sentences.pkl", 'wb'))

    ##################################################################### TODO: Only comment out if already ran code before!!!
    # test_sentences = pickle.load(open("test_sentences.pkl", 'rb')) # Uncomment this for dev file
    test_sentences = pickle.load(open("real_test_sentences.pkl", 'rb'))  # Uncomment this for test file
    #####################################################################
    print("Testing sentences extracted from file")

    m = len(test_sentences)
    print("m is: " + str(m))

    ##################### ADD-ONE SMOOTHING HERE ##########################
    add1_sum_perplexity = sum([get_add1_perplexity(s) for s in test_sentences])
    add1_test_perplexity = add1_sum_perplexity / m
    #######################################################################

    ##################### SIMPLE INTERPOLATION HERE #######################
    si_sum_perplexity = sum([get_si_perplexity(s) for s in test_sentences])
    si_test_perplexity = si_sum_perplexity / m
    #######################################################################

    print("\nAdd-1 Smoothing Perplexity is: " + str(add1_test_perplexity))
    print("Simple Interpolation Perplexity is: " + str(si_test_perplexity) + "\n")

    #*************** Generating 20 sentences from each LM *****************
    gen_sentences(20)


def get_add1_perplexity(sentence):
    global add_one_model
    """
    :param sentence: (String) one of the sentences
    :return: gives the add-one smoothing perplexity of particular sentence
    """
    accum = 1
    all_words = sentence.split()
    ######### Take out words that are introduced in testing set that are not in training set. ########
    all_words = ['UNK' if x not in unigram_model.keys() else x for x in all_words]
    all_trigrams = list(trigrams(all_words))

    ############################### N here is number of all possible trigrams ########################
    N = len(list(all_trigrams))
    # This is originally what the TA thought----
    # N = len(all_words)

    for w1, w2, w3 in all_trigrams:
        num = trigram_model[(w1, w2)][w3]
        den = sum(trigram_model[(w1, w2)].values())
        add_one_prob = (num + 1) / float(den + train_vocab_count)
        ########### Saving to a model #############
        add_one_model[(w1,w2)][w3] = add_one_prob
        ####### Important to calc perplexity ######
        accum *= add_one_prob
    perplexity = (1 / accum)** (1 / N)
    return perplexity


def get_si_perplexity(sentence):
    global si_model
    """
    :param sentence: (String) one of the sentences
    :return: gives the simple interpolation perplexity of particular sentence
    """
    l1 = 0.2 # TODO: These numbers can be continued to toggled based on dev set preferrences
    l2 = 0.7
    l3 = 0.1
    accum = 1
    all_words = sentence.split()
    ######### Take out words that are introduced in testing set that are not in training set. ########
    all_words = ['UNK' if x not in unigram_model.keys() else x for x in all_words]
    all_trigrams = list(trigrams(all_words))

    ############################### N here is number of all possible trigrams ########################
    N = len(list(all_trigrams))
    for w1, w2, w3 in all_trigrams:
        #~~~~~~~~ Building numerators ~~~~~~~~
        p1_num = trigram_model[(w1, w2)][w3]
        p2_num = bigram_model[w2][w3]
        p3_num = unigram_model[w3]
        # ~~~~~~~~ Building denominators ~~~~~
        p1_den = sum(trigram_model[(w1, w2)].values())
        p2_den = sum(bigram_model[w2].values())
        p3_den = sum(unigram_model.values())
        ########## Constructing parts ##########
        p1 =  p1_num / p1_den
        p2 = p2_num / p2_den
        p3 = p3_num / p3_den
        si_prob = l1 * p1 + l2 * p2 + l3 * p3
        ########### Saving to a model #############
        si_model[(w1, w2)][w3] = si_prob
        ####### Important to calc perplexity ######
        accum *= si_prob
    perplexity = (1 / accum)** (1 / N)
    return perplexity


def extract_text(file):
    """
    :param file: One of the input csv files
    :return: Save each line of the file to one large master string
    """
    df = pd.read_csv(file)
    saved_column = df.text
    master_string = ""
    for i in saved_column:
        # remove <laughter> and <<laughter>> as well as previous space w/ or w/o space after
        i = re.sub("[<<].*?[>>]", " ", i)
        # remove {F and {D w/ or w/o space after
        i = re.sub('\{[A-Z]', " ", i)
        # remove symbols /, +, #, -, *, ", and ending } with or without space after
        i = re.sub('[/+}#\-*"|]', " ", i)
        # remove all [...] brackets with or without space after
        i = re.sub('[\[\]]', " ", i)
        # remove all (( with or without space before
        i = re.sub('\(\(', " ", i)
        # remove all )) with or without space after
        i = re.sub('\)\)', " ", i)

        ######## Normalize all sentence punctuation so they = "word" #########
        i = re.sub('\.',' . ', i)
        i = re.sub(',', ' , ', i)
        i = re.sub('\?', ' ? ', i)
        i = re.sub('!', ' ! ', i)
        ######################################################################

        # remove double, triple, or more spaces -> single space
        i = re.sub(' +', ' ', i)
        # lower case all words
        i = i.lower()
        # add string beginners and enders, easier to split
        master_string += "|<s> <s> " + i.strip() + " </s>"
    return master_string



def training_work(training_string):
    """
    :param training_string: Given string after "extract_text"
    :return: List of cleaned strings (with UNK), list of words that are in UNK list
    """
    # Get all the vocab in the entire corpus
    vocab = set(training_string.split())
    print("Original vocab amount for training data: "+str(len(vocab))) # Value = 18156
    ##################################################################################
    training_tokens = training_string.split()

    # Get counts of all the vocab in the corpus (IOT elimate <5)
    vocab_counts = dict((v, training_tokens.count(v)) for v in vocab)
    less_than_five = list(k for k, v in vocab_counts.items() if v < 5)
    print("Total number of training vocab that is <5: "+str(len(less_than_five))) # Value = 12035

    # replace words <5 with 'UNK'
    training_tokens = ['UNK' if x in less_than_five else x for x in training_tokens]
    training_string = ' '.join(training_tokens)

    # new_vocab_count is old count subtracting the 'less than five'
    new_vocab_count = len(set(training_tokens))
    print("New vocab amount for training data: " +str(new_vocab_count)) # Value = 6121

    # Return all the sentences, without the first empty list
    return training_string.split('|')[1:], less_than_five, new_vocab_count


def testing_work(testing_string, less_than_five):
    """
    :param testing_string: Given string after "extract_text"
    :param less_than_five: List of unk from training string
    :return: List of cleaned strings (with UNK), vocab num from testing set
    """
    # Replace words <5 in training with 'UNK'
    testing_tokens = testing_string.split()
    testing_tokens = ['UNK' if x in less_than_five else x for x in testing_tokens]
    testing_string = ' '.join(testing_tokens)

    # Return all the sentences, without the first empty list
    return testing_string.split('|')[1:]


################################### For generating sentences from existing language models #############################

def gen_sentences(num):
    """
    :param num: Number of sentences needed to be generated from both models
    :return: Shows on terminal the sentences
    """
    for i in range(1, num+1):
        print("Add-One Sentence #" + str(i) + ": " + gen_add1_sentence())
    for i in range(1, num+1):
        print("SI Sentence #" + str(i) + ": " + gen_si_sentence())


def gen_add1_sentence():
    """
    :return: Creates a random sentence from add-one model
    """
    w1 = '<s>'
    w2 = '<s>'
    sentence = '<s> <s>'
    w3 = None
    while w3 != '</s>':
        poss_w3 = list(add_one_model[(w1,w2)].keys())
        probs = list(add_one_model[(w1,w2)].values())
        new_probs = [x / sum(probs) for x in probs]
        w3 = np.random.choice(poss_w3, 1, p=new_probs)[0]
        sentence += " " + w3
        w1 = w2
        w2 = w3
    return sentence


def gen_si_sentence():
    """
    :return: Creates a random sentence from simple-interpolation model
    """
    w1 = '<s>'
    w2 = '<s>'
    sentence = '<s> <s>'
    w3 = None
    while w3 != '</s>':
        poss_w3 = list(si_model[(w1,w2)].keys())
        probs = list(si_model[(w1,w2)].values())
        new_probs = [x / sum(probs) for x in probs]
        w3 = np.random.choice(poss_w3, 1, p=new_probs)[0]
        sentence += " " + w3
        w1 = w2
        w2 = w3
    return sentence


########### Called upon running function #########
main()

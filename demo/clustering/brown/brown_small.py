import nltk
import itertools
import math
from collections import defaultdict, Counter
import pickle
import os

k_init = 2

clusters_results_file = 'data/brownclusters-small.txt'

bg_N = 0 # Used for quality equation, N value of bigrams
unig_N = 0 # Used for quality equation, N value of unigrams

bigram_word_counts = defaultdict(lambda:0)
unigram_word_counts = defaultdict(lambda:0)
ordered_vocab = list()

binary_dictionary = defaultdict(lambda: '')


def main():
    global bg_N, unig_N, bigram_word_counts, ordered_vocab
    
    unigram_word_counts = {'the':6, 'house':4, 'is': 4, 'by':4, 'dog':2,'cat':2, 'that':2}
    corpus = ['the dog is by the house',
              'the cat is by the house',
              'the dog is by that house',
              'the cat is by that house']

    bigram_word_counts = get_bigrams(corpus)
    ordered_vocab = list(unigram_word_counts.keys())

    bg_N = sum([sum(x.values()) for x in bigram_word_counts.values()])
    unig_N = sum(unigram_word_counts.values())

    run_brown()        

    if not os.path.exists("./data"):
        os.makedirs('./data') 

    with open(clusters_results_file, 'w') as f:
        for word in binary_dictionary:
            f.write(word + '\t\t\t\t\t' + binary_dictionary[word] + '\n')
    f.close()


def load_corpus():
    """
    * Loads corpus file into data structure
    """
    with open(corpus_file, 'r') as f:
        corpus = f.readlines()
        return corpus

def load_ranked_vocab():
    """
    * Loads ranked vocab file into data structure
    """
    with open(vocab_file, 'r') as f:
        txt = f.readlines()
        data = [x.split() for x in txt]
        n = {x[0]:int(x[1]) for x in data}
        return n

def get_bigrams(corpus):
    """
    corpus : Clean corpus taken from file
    * Generates all bigrams from corpus (dictionary of dictionary)
    """
    bigram_model = defaultdict(lambda : defaultdict(lambda :0))
    for line in corpus:
        tokens = line.split()
        bgs = nltk.bigrams(tokens)
        fdist = nltk.FreqDist(bgs)
        for k,v in fdist.items():
            bigram_model[k[0]][k[1]] += v
    return bigram_model

def run_brown():
    global unigram_word_counts, bigram_word_counts
    cluster_to_word = list() # ordered list where index is cluster # to word
    word_to_cluster = dict()

    curr_unigram_counts = list()
    curr_bigram_counts = list()

    ################ ONLY DONE ONCE ##########################
    for i in range(k_init): # initialize C to be the top 200 words first
        curr_word = ordered_vocab.pop(0)

        word_to_cluster[curr_word] = i
        cluster_to_word.append([curr_word])
        curr_unigram_counts.append(unigram_word_counts[curr_word])
        curr_bigram_counts.append(bigram_word_counts[curr_word])

    for i in range(k_init):
        for j in list(curr_bigram_counts[i]):
            if j in word_to_cluster:
                curr_bigram_counts[i][word_to_cluster[j]] = curr_bigram_counts[i].pop(j)
            else:
                del curr_bigram_counts[i][j]

    ##########################################################
    for next_word in ordered_vocab:
        print("Adding word '" + next_word + "' into the mix")
        cluster_to_word.append([next_word])
        word_to_cluster[next_word] = k_init # new cluster (quality computer) after adding new word

        curr_unigram_counts.append(unigram_word_counts[next_word])
        next_bigram_word_counts = bigram_word_counts[next_word]

        for i in list(next_bigram_word_counts):
            if i in word_to_cluster:
                next_bigram_word_counts[word_to_cluster[i]] = next_bigram_word_counts.pop(i)
            else:
                del next_bigram_word_counts[i]

        curr_bigram_counts.append(next_bigram_word_counts)

        max_quality = 0
        arg_max = (0,0)

        #startingvals, starting_quality = get_quality(curr_unigram_counts, curr_bigram_counts)

        for first_i in list(range(k_init)):
            for second_i in list(range(first_i+1, k_init+1)):
                temp_unigram = curr_unigram_counts.copy()
                temp_bigram = curr_bigram_counts.copy()
                temp_cluster_to_word = cluster_to_word.copy()
                current_merge = (first_i, second_i)

                a = temp_unigram.pop(second_i)
                b = temp_unigram.pop(first_i)
                temp_unigram.insert(first_i, a+b)

                x = Counter(temp_bigram.pop(second_i))
                y = Counter(temp_bigram.pop(first_i))
                combined = dict(x+y)

                if second_i in combined:
                    if first_i in combined:
                        combined[first_i] = combined[first_i] + combined.pop(second_i)
                    else:
                        combined[first_i] = combined.pop(second_i)

                new_combined = dict()
                for x in list(combined):
                    if x > second_i:
                        new_combined[x - 1] = combined.pop(x)
                    else:
                        new_combined[x] = combined.pop(x)

                temp_bigram.insert(first_i, new_combined)

                # temp_word_to_cluster = {key:first_i if (word_to_cluster[key]==second_i or word_to_cluster[key]==first_i) else
                #                     word_to_cluster[key] for key in word_to_cluster.keys()} # make both indices the first index
                # temp_word_to_cluster = {key:(temp_word_to_cluster[key] - 1) if (temp_word_to_cluster[key] > second_i) else
                #                         temp_word_to_cluster[key] for key in temp_word_to_cluster.keys()} # lower all the rest of the indices after second index

                temp_cluster_to_word[first_i] = temp_cluster_to_word[first_i] + temp_cluster_to_word.pop(second_i)

                curr_quality = get_quality(temp_unigram, temp_bigram)

                if curr_quality > max_quality:
                    max_quality = curr_quality
                    arg_max = current_merge

                    best_cluster_to_word = temp_cluster_to_word
                    # best_word_to_cluster = temp_word_to_cluster
                    best_unigram_counts = temp_unigram
                    best_bigram_counts = temp_bigram
                    print("- - -  Quality replaced by: " + str(arg_max) + " with value " + str(max_quality))


        add_to_binary(arg_max, cluster_to_word) # This actually is the old cluster to word

        curr_unigram_counts = best_unigram_counts
        curr_bigram_counts = best_bigram_counts
        # word_to_cluster = best_word_to_cluster
        cluster_to_word = best_cluster_to_word

        print("Merged {0} and {1} together".format(arg_max[0], arg_max[1]))

    print("Done with merging to k_init clusters")

def add_to_binary(arg_max_tuple, cluster_to_word):
    global binary_dictionary

    first = arg_max_tuple[0]
    second = arg_max_tuple[1]

    for word in cluster_to_word[first]:
        print("Adding 0 to "+word)
        binary_dictionary[word] = '0' + binary_dictionary[word]

    for word in cluster_to_word[second]:
        print("Adding 1 to "+word)
        binary_dictionary[word] = '1' + binary_dictionary[word]

def get_quality(unigram, bigram):
    quality = 0
    #vallist = list()
    for first_cluster in range(len(bigram)):
        for second_cluster in bigram[first_cluster]:
            part1 = bigram[first_cluster][second_cluster] / bg_N
            part2 = unigram[first_cluster] / unig_N
            try:
                part3 = unigram[second_cluster - 1] / unig_N
            except IndexError:
                part3 = unigram[second_cluster] / unig_N
            val = part1 * math.log(part1 / (part2 * part3 + 1))

            # vallist.append(val)
            quality += val
    return abs(quality)

if __name__=="__main__":
    main()


from gensim import models
import os

results_file = './data/results-word2vec.txt'
data_file = 'GoogleNews-vectors-negative300.bin.gz' # Can be downloaded at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
topValue = 10
 
def main():
    model = models.KeyedVectors.load_word2vec_format(data_file, binary=True,
                                                     limit=200000) # 200,000 is the limit before I get Memory Error crash. Change according to your RAM.


    results = dict()
    input_word = ''
    words = list()
    print("Please provide words for word2vec. Empty means completion.")
    while input_word != '' or len(words)==0:
        input_word = input("Word: ")
        words.append(input_word)

    # Words being tested
    for word in words:
        try:
            data = model.most_similar(positive=[word],topn=topValue)
            results[word] = data
        except KeyError:
            results[word] = ['-']
            pass
    
    # Finally, returns results
    if not os.path.exists("./data"):
        os.makedirs('./data') 
    write_to_file(results)


def write_to_file(results):
    """
    results : organized results dictionary from word2vec model
    * Writes results to common format
    """
    with open(results_file, 'w') as f:
        for result in results:
            f.write("For word: {}\n".format(result))
            write_items = [x[0] + '\n' for x in results[result]]
            for i in write_items:
                f.write(i)
            f.write('\n')
    f.close()


if __name__=="__main__":
    main()
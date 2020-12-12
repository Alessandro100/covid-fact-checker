import csv
import math
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# No inputs
# Outputs a 2d array, [class1, class2][class_prior, class_score, class_score_filtered]
def create_vocabulary():
    word_dict = {}
    class_1_tweet_list = []
    class_2_tweet_list = []
    class_1_word_dict = {}
    class_2_word_dict = {}

    total_tweets = 0

    # finds the features to use for the Naive Bayes classifer
    with open("data/covid_training.tsv", encoding="utf-8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        next(rd)  # skips the header

        for row in rd:
            total_tweets = total_tweets + 1
            if(row[2] == "yes"):
                # gathers all tweets with verifiable factual claims (class1)
                class_1_tweet_list.append(row[1])
                # creates a list with all the words that are inside class 1
                class_1_text_list = row[1].split(" ")
                for word in class_1_text_list:
                    formatted_word = word.lower()
                    if formatted_word in class_1_word_dict:
                        # counts the word frequency inside class 1
                        class_1_word_dict[formatted_word] = class_1_word_dict[formatted_word] + 1
                    else:
                        class_1_word_dict[formatted_word] = 1

            else:
                # gathers all tweets with non-verifiable factual claims (class2)
                class_2_tweet_list.append(row[1])
                # creates a list with all the words that are inside class 2
                class_2_text_list = row[1].split(" ")
                for word in class_2_text_list:
                    formatted_word = word.lower()
                    if formatted_word in class_2_word_dict:
                        # counts the word frequency inside class 2
                        class_2_word_dict[formatted_word] = class_2_word_dict[formatted_word] + 1
                    else:
                        class_2_word_dict[formatted_word] = 1

            text_list = row[1].split(" ")
            for word in text_list:
                formatted_word = word.lower()
                if formatted_word in word_dict:
                    word_dict[formatted_word] = word_dict[formatted_word] + 1
                else:
                    word_dict[formatted_word] = 1

    class_1_filtered_word_dict = {key:val for key, val in class_1_word_dict.items() if val != 1}
    class_2_filtered_word_dict = {key:val for key, val in class_2_word_dict.items() if val != 1}
    
    original_vocabulary = []
    filtered_vocabulary = []
    class_1_vocabulary = []
    class_2_vocabulary = []
    class_1_vocabulary_filtered = []
    class_2_vocabulary_filtered = []

    # sorts the words into orginal / filtered
    # Don't really use vocabularies so might be able to delete
    for word in word_dict.keys():
        original_vocabulary.append(word)
        if(word_dict[word] > 1):
            filtered_vocabulary.append(word)

    for word in class_1_word_dict.keys():
        class_1_vocabulary.append(word)
        if(class_1_word_dict[word] > 1):
            class_1_vocabulary_filtered.append(word)

    for word in class_2_word_dict.keys():
        class_2_vocabulary.append(word)
        if(class_2_word_dict[word] > 1):
            class_2_vocabulary_filtered.append(word)
            
## Printing different stats
    print("Number of original vocabulary words")
    print(len(original_vocabulary))
    print("Number of filtered vocabulary words")
    print(len(filtered_vocabulary))
    print("Number of tweets in class 1")
    print(len(class_1_tweet_list))
    print("Number of original vocabulary words in class 1")
    print(len(class_1_vocabulary), sum(class_1_word_dict.values()))
    print("Number of filtered vocabulary words in class 1")
    print(len(class_1_vocabulary_filtered),  sum(class_1_filtered_word_dict.values()))
    print("Number of tweets in class 2")
    print(len(class_2_tweet_list))
    print("Number of original vocabulary words in class 2")
    print(len(class_2_vocabulary), sum(class_2_word_dict.values()))
    print("Number of filtered vocabulary words in class 2")
    print(len(class_2_vocabulary_filtered), sum(class_2_filtered_word_dict.values()) )
    print("Total number of tweets: ", total_tweets)


    class_1_prior = len(class_1_tweet_list) /(len(class_1_tweet_list) + len(class_2_tweet_list))
    class_2_prior = len(class_2_tweet_list) /(len(class_1_tweet_list) + len(class_2_tweet_list))
    
    score_class_1 = train_nb_classifier(class_1_word_dict,original_vocabulary)
    score_class_1_filtered = train_nb_classifier(class_1_filtered_word_dict, filtered_vocabulary)
    score_class_2 = train_nb_classifier(class_2_word_dict, original_vocabulary)
    score_class_2_filtered = train_nb_classifier(class_2_filtered_word_dict, filtered_vocabulary)
    
    nb_test(class_1_prior, score_class_1, class_1_word_dict,{1515155:['Trump', 'REEE', 'get', 'process', 'vaccine']}, original_vocabulary)
    class_1_info = [class_1_prior, score_class_1, score_class_1_filtered, class_1_word_dict, class_1_filtered_word_dict]
    class_2_info = [class_2_prior, score_class_2, score_class_2_filtered, class_2_word_dict, class_2_filtered_word_dict]
    return [class_1_info, class_2_info], original_vocabulary, filtered_vocabulary

# Inputs are class dictionnary {w: # of appearences}
# Output is the score of each word for the class {w: score_for_class}
def train_nb_classifier(class_dictionnary,vocabulary):
    scores = {}
    for w in class_dictionnary:
        score = math.log10((class_dictionnary[w]+0.01)/(sum(class_dictionnary.values())+len(vocabulary)*0.01))
        scores[w] = score
    return scores

# Inputs are the class prior (float), class scores { word : class_score },class_dictionnary,tweet as list of word ie['get', 'me', 'out']
# Output is score of class for tested tweet
def nb_test(class_prior, class_scores, class_dictionnary, tweet, vocabulary):
    score = math.log10(class_prior)
    for word in tweet:
        if word in vocabulary:
            if word in class_scores:
                score += class_scores[word] 
            else:
                score += math.log10(0.01/(sum(class_dictionnary.values())+len(vocabulary)*0.01))
    return score

# No inputs
# Output is a dict of tweets of words ie {1232312651: ['This', 'is', 'a', 'tweet', 'yeah', 'dude']}
def get_testing_set():
    tokenised_tweets= {}
    with open("data/covid_test_public.tsv", encoding="utf-8") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            tokenised_tweets[row[0]] = {'words': row[1].split(" "), 'correct_class': row[2]}
    return tokenised_tweets

def trace_to_file(file_name, tweets):
    with open(file_name, 'w') as f:
        for tweet in tweets:
            f.write(tweet['id'] + '  ' +
                    tweet['predicted_class'] + '  ' +
                    "{:.2e}".format(tweet['score']) + '  ' +
                    tweet['correct_class'] + '  ' +
                    tweet['outcome'] + '\n')

def get_performance(file_name, tweets):
    tweet_correct = np.array([])
    tweet_predicted = np.array([])
    true_postives = 0

    for tweet in tweets:
        tweet_correct = np.append(tweet_correct, tweet['correct_class'])
        tweet_predicted = np.append(tweet_predicted, tweet['predicted_class'])
        if (tweet['correct_class'] == tweet['predicted_class']):
            true_postives = true_postives + 1

    # precision: tp/tp+fp, recall: tp/tp+fn, f1: 2*precision*recall/precision+recall
    stats = precision_recall_fscore_support(tweet_correct, tweet_predicted, average = None, labels = ['yes', 'no'])
    accuracy = true_postives/len(tweets)
    
    print(accuracy, stats)


    with open(file_name, 'w') as f:
        f.write(str(accuracy) + '\n' +
                str(stats[0][0]) + '  ' + str(stats[0][1]) + '\n' +
                str(stats[1][0]) + '  ' + str(stats[1][1]) + '\n' +
                str(stats[2][0]) + '  ' + str(stats[2][1]) + '\n')
    
def main():
    
    tokenised_tweets_to_classify = get_testing_set() 
    class_infos, original_vocabulary, filtered_vocabulary = create_vocabulary()  

    original_results = []
    filtered_results = []
    
    for tweet in tokenised_tweets_to_classify:
        class1_score = nb_test(class_infos[0][0], class_infos[0][1], class_infos[0][3], tokenised_tweets_to_classify[tweet]['words'], original_vocabulary)
        class2_score = nb_test(class_infos[1][0],class_infos[1][1], class_infos[1][3], tokenised_tweets_to_classify[tweet]['words'], original_vocabulary)
        
        is_verifiable = False
        if (class1_score > class2_score ):
            is_verifiable = True        

        predicted_class = 'yes' if is_verifiable else 'no'
        correct_class = tokenised_tweets_to_classify[tweet]['correct_class']

        original_results.append({
            'id': tweet,
            'predicted_class': predicted_class,
            'score': class1_score if is_verifiable else class2_score,
            'correct_class': correct_class,
            'outcome': 'correct' if predicted_class == correct_class else 'wrong'
        })
    
    for tweet in tokenised_tweets_to_classify:
        class1_score = nb_test(class_infos[0][0], class_infos[0][2], class_infos[0][4], tokenised_tweets_to_classify[tweet]['words'], filtered_vocabulary)
        class2_score = nb_test(class_infos[1][0],class_infos[1][2], class_infos[1][4], tokenised_tweets_to_classify[tweet]['words'], filtered_vocabulary)
        
        is_verifiable = False
        if (class1_score > class2_score ):
            is_verifiable = True  

        predicted_class = 'yes' if is_verifiable else 'no'
        correct_class = tokenised_tweets_to_classify[tweet]['correct_class']

        filtered_results.append({
            'id': tweet,
            'predicted_class': predicted_class,
            'score': class1_score if is_verifiable else class2_score,
            'correct_class': correct_class,
            'outcome': 'correct' if predicted_class == correct_class else 'wrong'
        })

    
    #outputs
    print("Original Vocabulary")      
    trace_to_file('trace NB-BOW-OV.txt', original_results)
    get_performance('eval NB-BOW-OV.txt', original_results)
    
    print("Filtered Vocabulary")      
    trace_to_file('trace NB-BOW-FV.txt', filtered_results)
    get_performance('eval NB-BOW-FV.txt', filtered_results)


main()


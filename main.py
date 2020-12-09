import csv
import math

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
    class_1_info = [class_1_prior, score_class_1, score_class_1_filtered, class_1_word_dict, class_1_filtered_word_dict,original_vocabulary]
    class_2_info = [class_2_prior, score_class_2, score_class_2_filtered, class_2_word_dict, class_2_filtered_word_dict, filtered_vocabulary]
    return [class_1_info, class_2_info]

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
            tokenised_tweets[row[0]] = row[1].split(" ")
    return tokenised_tweets
    
def main():
    
    tokenised_tweets_to_classify = get_testing_set() 
    class_infos = create_vocabulary()   
    print(tokenised_tweets_to_classify)
    
    for tweet in tokenised_tweets_to_classify:
       if(nb_test(class_infos[0][0],class_infos[0][1],  class_infos[0][3],tokenised_tweets_to_classify[tweet], class_infos[0][5] ) > nb_test(class_infos[1][0],class_infos[1][1], class_infos[1][3], tokenised_tweets_to_classify[tweet], class_infos[0][5])):
           print("Tweet # ", tweet, "classified as verifiable factual information",nb_test(class_infos[0][0],class_infos[0][1], class_infos[0][3], tokenised_tweets_to_classify[tweet], class_infos[0][5]), "vs ", nb_test(class_infos[1][0],class_infos[1][1], class_infos[1][3] ,tokenised_tweets_to_classify[tweet], class_infos[0][5]) )
       else: 
           print("Tweet # ", tweet, "classified as non-verifiable factual information",nb_test(class_infos[0][0],class_infos[0][1], class_infos[0][3], tokenised_tweets_to_classify[tweet], class_infos[0][5]), "vs ", nb_test(class_infos[1][0],class_infos[1][1], class_infos[1][3] ,tokenised_tweets_to_classify[tweet], class_infos[0][5]) )
    print("\n now for filtered vocabulary \n")      
    for tweet in tokenised_tweets_to_classify:
       if(nb_test(class_infos[0][0],class_infos[0][2],  class_infos[0][4],tokenised_tweets_to_classify[tweet], class_infos[1][5]) > nb_test(class_infos[1][0],class_infos[1][2], class_infos[1][4],tokenised_tweets_to_classify[tweet], class_infos[1][5])):
           print("Tweet # ", tweet, "classified as verifiable factual information",nb_test(class_infos[0][0],class_infos[0][2], class_infos[0][4],tokenised_tweets_to_classify[tweet], class_infos[1][5]), "vs ", nb_test(class_infos[1][0],class_infos[1][2], class_infos[1][4],tokenised_tweets_to_classify[tweet], class_infos[1][5]) )
       else: 
           print("Tweet # ", tweet, "classified as non-verifiable factual information",nb_test(class_infos[0][0],class_infos[0][1], class_infos[0][3], tokenised_tweets_to_classify[tweet], class_infos[0][5]), "vs ", nb_test(class_infos[1][0],class_infos[1][1], class_infos[1][3] ,tokenised_tweets_to_classify[tweet], class_infos[0][5]) )
        

main()


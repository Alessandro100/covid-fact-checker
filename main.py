import csv

# NOTE
# The words that are placed in the dictionary are all lowercase

word_dict = {}

# finds the features to use for the Naive Bayes classifer
with open("data/covid_training.tsv") as fd:
    print('start')
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        text_list = row[1].split(" ")
        for word in text_list:
            formatted_word = word.lower()
            if formatted_word in word_dict:
                word_dict[formatted_word] = word_dict[formatted_word] + 1
            else:
                word_dict[formatted_word] =  1


original_vocabulary = []
filtered_vocabulary = []

# sorts the words into orginal / filtered
for word in word_dict.keys():
    original_vocabulary.append(word)
    if(word_dict[word] > 1):
        filtered_vocabulary.append(word)

print("Number of original vocabulary words")
print(len(original_vocabulary))
print("Number of filtered vocabulary words")
print(len(filtered_vocabulary))

# original_vocabulary contains the list of all words (features to use) in a list data structure
# filtered_vocabulary contains the list of filtered words (features to use) in a list data structure
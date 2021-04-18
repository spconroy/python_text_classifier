import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from joblib import dump, load

from sklearn import tree
import numpy


#Create empty dictionary
word_dictionary = []
max_sentencelength = 100

#Function to Add words in sentence to word dictionary
def populate_dictionary(text):
    for word in word_tokenize(text):
        if word not in word_dictionary:
            word_dictionary.append(word)

def vectorize_text(text):
    vector_array = []
    for word in word_tokenize(text):
        vector_array.append(word_dictionary.index(word))
    while len(vector_array) < max_sentencelength:
        vector_array.append(0)
    return vector_array
        

sentence_list = ["I am happy about my test results","I am sad about my test results","We did great at the game yesterday","I Cried all night after I failed","I failed the big test last weekend","We played the game well on tuesday"]

#Add all words in all sentences to dictionary
for sentence in sentence_list:
    populate_dictionary(sentence)

vec_sentence_list = []
for sentence in sentence_list:
    vec_sentence_list.append(vectorize_text(sentence))


#labels: -1 for negative sentiment. 1 for positive sentiment
text_labels = [1,-1,1,-1,-1,1]

#Declare classifier object
clf = tree.DecisionTreeClassifier()
 
#Feed in Training Data
clf.fit(vec_sentence_list,text_labels)


test_sentence1 = "I am happy I did well yesterday"
test_sentence2 = "I am sad about the big test"

result = clf.predict(numpy.array(vectorize_text(test_sentence1)).reshape(1,-1))
#result = clf.predict(vectorize_text(test_sentence1))

if result == 1:
    print("Happy")

if result == -1:
    print("Sad")

print(result)

#save model and dictionary to disk
dump(clf, 'model.joblib')
dump(word_dictionary, 'dictionary.joblib')


#Load model and dictionary from disk
word_dictionary = load('dictionary.joblib')
clf = load('model.joblib')

test_sentence1 = "I am happy I did well yesterday"
test_sentence2 = "I am sad about the big test"

result = clf.predict(numpy.array(vectorize_text(test_sentence2)).reshape(1,-1))
#result = clf.predict(vectorize_text(test_sentence1))

if result == 1:
    print("Happy")
    print(result)

if result == -1:
    print("Sad")
    print(result)

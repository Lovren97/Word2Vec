#import enchant
import xml.etree.ElementTree as ET
import sys
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
numpy.set_printoptions(threshold=sys.maxsize)
import gensim.models.keyedvectors as word2vec

"-------------------Loading Google's Word2Vec model--------------"
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = word2vec.KeyedVectors.load_word2vec_format('/storage/GoogleNews-vectors-negative300.bin', binary=True)
"----------------------------------------------------------------"

"--------------------Dictonary definitions------------------------"
cachedStopWords = stopwords.words("english")
#english_dictonary = enchant.Dict("en_US")
"-----------------------------------------------------------------"


"--------------------Parameters definitions-----------------------"
#articles
tree = ET.parse("articles/first40.xml")
all_articles = tree.getroot()

#truths
oak_tree = ET.parse("D:\\Files\\truth10.xml")
all_truth = oak_tree.getroot()


#here we are saving all words ready for vectorizing
article_words = []

#here we are saving all those currently unused stuff like title,date,url
#and etc.
article_unused = []

word2vecs = []
"------------------------------------------------------------------"



"-------------------XML parsing and data mining--------------------"
#going through every article in xml file
for article in all_articles:
    article_unused.append(article.attrib)
    article_text = ""
    paragraph_words = []
    for paragraph in article:
        if(paragraph.text != None):
            words = paragraph.text
            words = words.split(" ")
            for word in words:
                if word not in cachedStopWords:
                    if(len(word) > 0):
                        #if(english_dictonary.check(word)):
                        if(word[-1] == '.' or word[-1] == '!' or word[-1] == '?'):
                            word = word[0:-1]
                        paragraph_words.append(word)
    article_words.append(paragraph_words)
"------------------------------------------------------------------"  

"--------------------Word2Vec modeling------------------------------"
for paragraph in article_words:
    paragraph_vectors = []
    for word in paragraph:
        if(word in model.vocab):
            vector = model[word]
            paragraph_vectors.append(vector)
        
    
    sum = paragraph_vectors[0]
    for i in range(1,len(paragraph_vectors)):
        sum = sum+paragraph_vectors[i]
    word2vecs.append(sum)

"--------------------------------------------------------------------"
    
"--------------------DATA OUTPUT-------------------------------------"
with open('truth.txt', 'w') as f:
    for vector in word2vecs:
        for item in vector:
            f.write("%s\n" % item)
"--------------------------------------------------------------------"

"-----------------------Truth processing--------------------------"

#final vector for ML testing
truth_vector = []

#going through every truth in xml file
for truth in all_truth:
    current_truth = truth.attrib
    hyperpartisan = current_truth['hyperpartisan']
    if(hyperpartisan == 'true'):
        truth_vector.append(1)
    else:
        truth_vector.append(0)

with open('truth.txt', 'w') as f:
    for item in truth_vector:
        f.write("%s\n" % item)

"------------------------------------------------------------------"


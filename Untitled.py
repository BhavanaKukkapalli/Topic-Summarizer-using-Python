#!/usr/bin/env python
# coding: utf-8

# In[188]:
import nltk
# In[189]:
nltk.download('punkt')
# In[190]:
nltk.download('stopwords')
# In[191]:
import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 
# In[194]:
df=pd.read_csv('new1.csv')
#df.head()
# In[220]:
from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list
sentences=list(set(sentences))
#sentences = [y for x in sentences for y in x] # flatten list
# In[221]:
len(sentences)
# In[222]:
# remove punctuations, numbers and special characters
sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
# In[223]:
# make alphabets lowercase
sentences = [s.lower() for s in sentences]
sentences
# In[224]:
nltk.download('averaged_perceptron_tagger')
# In[225]:
temp_list=[]
for i in sentences: 
      
    # Word tokenizers is used to find the words  
    # and punctuation in a string 
    wordsList = nltk.word_tokenize(i) 
  
    # removing stop words from wordList 
#     wordsList = [w for w in wordsList if not w in stop_words]  
  
    #  Using a Tagger. Which is part-of-speech  
    # tagger or POS-tagger.  
    tagged = nltk.pos_tag(wordsList) 
    temp_list.append(tagged)
#     print(tagged) 
temp_list
# In[226]:
def findNN(list):
    for i in range(0,len(list)):
        if(list[i][1]=="NN"):
            return list[i][0]
        
def isNN(list):
    for i in range(0,len(list)):
        if(list[i][1]=="NN"):
            return 1
    return 0

def isPR(list):
    for i in range(0,len(list)):
        if((list[i][1]=="PRP") or (list[i][1]=="PRP$")):
            return 1
    return 0

def findPRP(list,i):
    for p in range(0,len(list)):
        if((list[p][1]=="PRP") or (list[p][1]=="PRP$")):
            return p


# In[202]:
prev_noun=" "
is_n=False
is_p=False
# temp_list
for i in range(0,len(temp_list)):
    if(isNN(temp_list[i])):
        noun=findNN(temp_list[i])
        is_n=True
        prev_noun=noun
    if(isPR(temp_list[i])):
        a=i;
        b=findPRP(temp_list[i],i)
        is_p=True
    if(is_n==True and is_p==True):
        temp=list(temp_list[a][b]).copy()
#         print(temp)
#         print(temp_list[a])
        temp[0]=noun;
#         print(temp_list[a])
        temp_list[a][b]=tuple(temp);
#         print(temp_list[a])
    if(is_n==False and is_p==True):
        temp=list(temp_list[a][b]).copy()
#         print(temp)
#         print(temp_list[a])
        temp[0]=prev_noun;
#         print(temp_list[a])
        temp_list[a][b]=tuple(temp);
#         print(temp_list[a])
#         temp_list[a][b][0]=prev_noun
    is_n=False
    is_p=False
        
temp_list


# In[227]:
p=0;
final=[]
for i in range(0,len(temp_list)):
    s=[]
    for j in range(0,len(temp_list[i])):
        s.append(temp_list[i][j][0])
    final.append(" ".join(s))
    p=p+1
len(final)
final
# In[228]:
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# In[229]:
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
# remove stopwords from the sentences
final = [remove_stopwords(r.split()) for r in final]
final
# In[206]:
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove*.zip
# In[ ]:
# In[230]:
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
# In[231]:
sentence_vectors = []
for i in final:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)
# In[232]:
len(sentence_vectors)
# In[233]:
sim_mat = np.zeros([len(final), len(final)])
sim_mat1 = np.zeros([len(final), len(final)])
sim_mat2 = np.zeros([len(final), len(final)])
# In[234]:
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(final)):
    for j in range(len(final)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
sim_mat
# In[235]:
import networkx as nx
# In[236]:
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
# In[237]:
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 10 sentences as the summary
for i in range(15):
    print(ranked_sentences[i][1])
# In[238]:

# from sklearn.metrics.pairwise import euclidean_distances
# for i in range(len(final)):
#     for j in range(len(final)):
#         if i != j:
#             sim_mat1[i][j] = euclidean_distances(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
# sim_mat1


# # In[239]:


# nx_graph1 = nx.from_numpy_array(sim_mat1)
# scores1 = nx.pagerank(nx_graph1)


# # In[240]:


# ranked_sentences1 = sorted(((scores1[i],s) for i,s in enumerate(sentences)), reverse=True)
# # Extract top 10 sentences as the summary
# for i in range(15):
#     print(ranked_sentences1[i][1])


# # In[241]:


# from sklearn.metrics.pairwise import manhattan_distances
# for i in range(len(final)):
#     for j in range(len(final)):
#         if i != j:
#             sim_mat2[i][j] = manhattan_distances(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
# sim_mat2


# # In[242]:


# nx_graph2 = nx.from_numpy_array(sim_mat2)
# scores2 = nx.pagerank(nx_graph2)


# # In[243]:


# ranked_sentences2 = sorted(((scores2[i],s) for i,s in enumerate(sentences)), reverse=True)
# # Extract top 10 sentences as the summary
# for i in range(15):
#     print(ranked_sentences2[i][1])


# # In[ ]:





# # In[ ]:





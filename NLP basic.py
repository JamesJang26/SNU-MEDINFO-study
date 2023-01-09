#!/usr/bin/env python
# coding: utf-8

# In[7]:


### Text preprocessing


# In[8]:


## Tokenization


# In[9]:


pip install nltk


# In[10]:


conda install tensorflow


# In[11]:


conda update -n base -c defaults conda


# In[12]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# In[13]:


print('단어 토큰화1 :', word_tokenize("Don't be fooled by the dark sounding name, Mr.Jones's Orphanage is as cheery as cheery goes for a pastry shop."))


# In[14]:


print('단어 토큰화2 :', WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr.Jones's Orphanage is as cheery as cheery goes for a pastry shop."))


# In[15]:


from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. It doesn't have a food chain or restaurant of their own."
print('Treebank wordtokenizer :', tokenizer.tokenize(text))


# In[16]:


# Sentence tokenization
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :', sent_tokenize(text))


# In[17]:


# 한국어 문장 토큰화 : korean sentence splitter
pip install kss
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :', kss.split_sentences(text))


# In[19]:


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화: ', tokenized_sentence)
print('품사 태깅: ', pos_tag(tokenized_sentece))


# In[21]:


import re
text = "I was wondering if anyone out there could enlighten me on this car."

shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))


# In[ ]:


# Stemming and Lemmatization


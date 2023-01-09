#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Text preprocessing


# In[2]:


## Tokenization


# In[3]:


# pip install nltk


# In[4]:


# conda install tensorflow


# In[5]:


# conda update -n base -c defaults conda


# In[6]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# In[7]:


print('단어 토큰화1 :', word_tokenize("Don't be fooled by the dark sounding name, Mr.Jones's Orphanage is as cheery as cheery goes for a pastry shop."))


# In[8]:


print('단어 토큰화2 :', WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr.Jones's Orphanage is as cheery as cheery goes for a pastry shop."))


# In[9]:


from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. It doesn't have a food chain or restaurant of their own."
print('Treebank wordtokenizer :', tokenizer.tokenize(text))


# In[10]:


# Sentence tokenization
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :', sent_tokenize(text))


# In[11]:


# 한국어 문장 토큰화 : korean sentence splitter

import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :', kss.split_sentences(text))


# In[13]:


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화: ', tokenized_sentence)
print('품사 태깅: ', pos_tag(tokenized_sentence))


# In[14]:


import re
text = "I was wondering if anyone out there could enlighten me on this car."

shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))


# In[15]:


# Stemming and Lemmatization

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :', words)
print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in words])


# In[16]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

sentence = "This was not the map we found in Billy Bone's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence = word_tokenize(sentence)

print('어간 추출 전 :', tokenized_sentence)
print('어간 추출 후 :', [stemmer.stem(word) for word in tokenized_sentence])


# In[17]:


# stopwords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from konlpy.tag import Okt
# kss 설치 안되서 그런듯?


# In[18]:


stop_words_list = stopwords.words('english')
print('불용어 개수 :', len(stop_words_list))
print('불용어 10개 출력 :', stop_words_list[:10])


# In[19]:


example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result = []
for word in word_tokens:
    if word not in stop_words:
        result.append(word)
        
print('불용어 제거 전 :', word_tokens)
print('불용어 제거 후 :', result)


# In[20]:


# regular expression
# 직접 예제들 해봐야하나? 찾아가면서 쓰면 될 것 같은데


# In[21]:


# Integer encoding

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

sentences = sent_tokenize(raw_text)
print(sentences)


# In[22]:


vocab = {} # 얘는 무슨 이유로 있는거지? 왜 dictionary로 받는거?
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    result = []
    
    for word in tokenized_sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)
print(preprocessed_sentences)


# In[23]:


print('단어 집합:', vocab) # print(vocab) 하면 안되는 이유?


# In[24]:


vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) # lambda가 머임?
print(vocab_sorted)


# In[25]:


word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency > 1:
        i = i + 1
        word_to_index[word] = i
        
print(word_to_index)


# In[26]:


vocab_size = 5 # 이렇게 크기 설정? 그게 아니라 뒤에 for 문에서 사용하기 위한 값으로 설정

words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size +1]

for w in words_frequency:
    del word_to_index[w]
print(word_to_index)


# In[27]:


word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)


# In[28]:


encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)


# In[29]:


from collections import Counter
print(preprocessed_sentences)


# In[30]:


all_words_list = sum(preprocessed_sentences, [])
print(all_words_list)


# In[31]:


vocab = Counter(all_words_list)
print(vocab) # 위에서 그냥 print(vocab)은 안됐는데 여기선 되는 이유? Counter처리해줘서 차이 있는거?


# In[32]:


vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)


# In[33]:


word_to_index = {}
i = 0
for (word, frequency) in vocab:
    i = i + 1
    word_to_index[word] = i
    
print(word_to_index)


# In[34]:


from nltk import FreqDist
import numpy as np

vocab = FreqDist(np.hstack(preprocessed_sentences)) # np.hstack
print(vocab['barber'])


# In[35]:


vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)


# In[36]:


word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)


# In[37]:


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)

print('인덱싱: ', tokenizer.word_index)
print('카운트: ', tokenizer.word_counts)


# In[38]:


# padding

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)


# In[39]:


max_len = max(len(item) for item in encoded)
print('최대 길이 :', max_len)


# In[40]:


for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)
        
padded_np = np.array(encoded)
print(padded_np)


# In[41]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
#padded = pad_sequences(encoded)
#print(padded)
padded = pad_sequences(encoded, padding = 'post')
print(padded)


# In[42]:


(padded == padded_np).all()


# In[43]:


# one-hot encoding


# In[44]:


# conda install pandas


# In[45]:


# conda install scikit-learn


# In[46]:


# splitting data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X, y = zip(['a', 1], ['b', 2], ['c', 3])
sequences = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequences)
print(['X 데이터 :', X])
print(['y 데이터 :', y])


# In[47]:


values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns = columns)
print(df)


# In[48]:


X = df['메일 본문']
y = df['스팸 메일 유무']
print('X 데이터 :', X.to_list())
print('y 데이터 :', y.to_list())


# In[49]:


np_array = np.arange(0, 16).reshape((4,4))
print('전체 데이터 :')
print(np_array)


# In[50]:


# BoW

from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)
    
    word_to_index = {}
    bow = []
    
    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            bow.insert(len(word_to_index) - 1, 1)
        else:
            index = word_to_index.get(word)
            bow[index] = bow[index] + 1
            
    return word_to_index, bow


# In[51]:


doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)
print('vocabulary :', vocab)
print('bag of words vector :', bow)


# In[52]:


doc2 = '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'

vocab, bow = build_bag_of_words(doc2)
print('vocabulary :', vocab)
print('bag of words vector :', bow)


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. Because I love you.']
vector = CountVectorizer()

print('bag of words vector :', vector.fit_transform(corpus).toarray())
print('vocabulary :', vector.vocabulary_) # 함수 자체가 vector.vocabulary_ 로 생긴거임?


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text = ["Family is not an important thing. It's everything."]
stop_words = stopwords.words('english')
vect = CountVectorizer(stop_words = stop_words)
print('bag of words vector :', vect.fit_transform(text).toarray())
print('vocabulary :', vect.vocabulary_)


# In[55]:


import pandas as pd
from math import log

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()


# In[56]:


N = len(docs)

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df+1))

def tfidf(t, d):
    return tf(t, d) * idf(t)


# In[58]:


result = []

for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))
        
tf_ = pd.DataFrame(result, columns = vocab)
print(tf_)


# In[8]:


result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))
    
idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
print(idf_)


# In[9]:


result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t,d))
        
tfidf_ = pd.DataFrame(result, columns = vocab)
print(tfidf_)


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray())
print(vector.vocabulary_)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)


# In[ ]:





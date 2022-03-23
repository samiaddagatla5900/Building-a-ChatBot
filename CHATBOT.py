#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer


# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD


# In[4]:


lemmatizer=WordNetLemmatizer()


# In[5]:


with open('Downloads\\intents (2).json') as json_data:
    intents = json.load(json_data)


# In[6]:


intents


# In[7]:


words = []   #patterns
classes = []   #tag
documents = []    #combinations
ignore_letters = ['?','!','.',',']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
            
print(documents)


# In[8]:


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words)


# In[9]:


classes = sorted(set(classes))
print(classes)


# In[10]:


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# In[11]:


training = []
output_empty = [0] * len(classes)


# In[12]:


for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


# In[13]:


random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])


# In[14]:


model=Sequential()
model.add(Dense(3000,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))


# In[15]:


model.compile(optimizer='sgd',
             loss='categorical_crossentropy',
             metrics = ['accuracy'])

hist=model.fit(np.array(train_x),np.array(train_y),epochs=500)


# In[16]:


model.save("chatbotmodel.hS",hist)


# In[17]:


get_ipython().system('pip install colorama')


# In[18]:


import random
import json
import pickle
import numpy as np

import colorama 
colorama.init()
from colorama import Fore, Style, Back


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model


# In[19]:


lemmatizer=WordNetLemmatizer()
with open('Downloads\\intents (2).json') as json_data:
    intents = json.load(json_data)


# In[20]:


words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbotmodel.hS')


# In[ ]:


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow])) [0]
    ERROR_THRESHOLD = 0.25
    results =[[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability': str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
    

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)   


while True:
    message = input("You-")
    if message.lower() =="quit":
       break
    ints = predict_class(message)
    res = get_response(ints,intents)


# In[ ]:





# In[ ]:





# In[ ]:






# coding: utf-8

#  <챗봇 Miner: Text Mining>  
#      
# - <b><a href='#the_destination'>1.DB 연동</a></b>
#     - <a href='#the_destination1'>1.1 DB 불러오기</a>
#     - <a href='#the_destination2'>1.2 DB 인덱스 수정</a>
#     
#     
# - <b><a href='#the_destination3'>2. Text Mining - 품사별 어휘 분석</a></b>
#     - <a href='#the_destination4'>2.1 텍스트 불러오기</a>    
#     - <a href='#the_destination5'>2.2 명사</a>
#         + <a href='#the_destination6'>2.2.1 명사 추출</a>
#             - <a href='#the_destination7'>A. 한 자리수 이상 명사 추출</a>           
#         - <a href='#the_destination8'>2.2.2 명사 어휘 빈도 및 그래프</a>
#             - <a href='#the_destination9'>A. MATPLOTLIB 그래프</a>
#             - <a href='#the_destination10'>B. 명사, 빈도수 추출</a>
#             - <a href='#the_destination11'>C. 명사, 빈도수 데이터 DataFrame 형식으로 변환</a>
#             - <a href='#the_destination12'>D. PLOTLY 그래프</a>
#             - <a href='#the_destination13'>E. 워드클라우드</a>
#     - 2.3 형용사, 동사
#         - 2.3.1 형용사, 동사 어휘 추출
# - <b><a href='#the_destination16'>3. Text Mining - 연관성 분석</b>
#     - <a href='#the_destination17'>3.1 KOMORAN
#         - <a href='#the_destination18'>3.1.1KOMORAN 형태소 분석
#         - <a href='#the_destination19'>3.1.2 KOMORAN 명사 추출
#         - <a href='#the_destination20'>3.1.3 KOMORAN 명사 빈도수 추출
#         - <a href='#the_destination21'>3.1.4 KOMORAN 시각화
#         - <a href='#the_destination22'>3.1.5 Unique한 명사 리스트 만들기
#         - <a href='#the_destination23'>3.1.6 문장-단어 행렬
#         - <a href='#the_destination24'>3.1.7 공존 단어 행렬 계산
#         - <a href='#the_destination25'>3.1.8 네트워크 그래프
#     - <a href='#the_destination26'>3.2 TWITTER
#         - <a href='#the_destination27'>3.2.1 TWITTER 형태소 분석
#     - <a href='#the_destination28'>3.2.2 TWITTER 명사 추출
#     - <a href='#the_destination29'>3.2.3 TWITTER 명사 빈도수 추출
#     - <a href='#the_destination30'>3.2.4 TWITTER 시각화
#     - <a href='#the_destination31'>3.2.5 Unique한 명사 리스트 만들기
#     - <a href='#the_destination32'>3.2.6 문자-단어 행렬
#     - <a href='#the_destination33'>3.2.7 공존-단어 행렬 계산
#     - <a href='#the_destination34'>3.2.8 네트워크 그래프
# - <b>4. 감성분석</b>
#     - <a href='#the_destination34'>4.1긍부정 트랜드 출력

# <a id='the_destination'></a>
# 
# # 1.DB 연동
# - 1.1 DB 불러오기
# - 1.2 DB 인덱스 수정

# <a id='the_destination1'></a>
# ## 1.1 DB 불러오기

# In[45]:

import pymysql.cursors
import numpy as np
conn = pymysql.connect(host='169.56.124.93', user='airchat' , password='airchat1234', charset='utf8')
curs = conn.cursor(pymysql.cursors.DictCursor) # Connection 으로부터 Dictoionary Cursor 생성

sql='SELECT CHAT_CNVRS_ID as ID,substr(CHAT_SEND_DTS,1,14) as date, substr(CHAT_SEND_DTS,1,14) as date_index ,CHAT_SEND_TEXT as text, CHAT_CONFI_RATE as conf from airchat.ICHAT_LOG where CHAT_CONFI_RATE > 0   '
a=curs.execute(sql)#쿼리문에 의해 디비를 불러옴

db=curs.fetchall()
#print(float(a))

#rows=curs.fetchall()

#avg=np.mean(rows)
#print(rows)

conn.close()


# In[46]:

import pandas as pd
from pandas import Series, DataFrame
db1=DataFrame(db)

db1['datetime']=db1['date_index'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d%H%M%S'))  #db1에 datetime 이라는 index를 설정해주기 위해 datetime 이라는 열을 설정
#  .apply(lambda x: ~~~ 의 의미는  내가 x를 다룰 건데, ~~~ 이런식으로 할꺼야 라는 뜻
# %Y%m%d 형식으로 된 X를 pandas의 to_datetime 함수를 통해 datetime object로 변환
db1['message_num']=1   #메세지 수를 합산하기 위해 만든 컬럼
db1.set_index(db1['datetime'], inplace=True)  #datetime 컬럼을 index로 만듬 

db1=db1.drop('datetime',1)  #기존에 만들었던 datetime 컬럼을 삭제
db1=db1.drop('date_index', 1)
db1=db1.drop('date', 1)
db1


# <a id='the_destination2'></a>
# ## 1.2 DB 인덱스 수정

# In[47]:

import pymysql.cursors
import numpy as np
conn = pymysql.connect(host='169.56.124.93', user='airchat' , password='airchat1234', charset='utf8')
curs = conn.cursor(pymysql.cursors.DictCursor) # Connection 으로부터 Dictoionary Cursor 생성


sql='select CHAT_CNVRS_ID as ID, timestampdiff(second, FRST_CONN_DTM, LAST_CONN_DTM) as stay_sec, FRST_CONN_DTM as date from airchat.ICHAT_CONN_STAT'
#sql='SELECT DISTINCT CHAT_CNVRS_ID as ID ,substr(CHAT_SEND_DTS,1,8) as date from chat.ICHAT_LOG where CHAT_CONFI_RATE > 0  '
a=curs.execute(sql)#쿼리문에 의해 디비를 불러옴

con_ID=curs.fetchall()
#print(float(a))

#rows=curs.fetchall()

#avg=np.mean(rows)
#print(rows)

conn.close()


# In[54]:

con_ID1=DataFrame(con_ID)

con_ID1['datetime']=con_ID1['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d%H%M%S'))  #db1에 datetime 이라는 index를 설정해주기 위해 datetime 이라는 열을 설정
#  .apply(lambda x: ~~~ 의 의미는  내가 x를 다룰 건데, ~~~ 이런식으로 할꺼야 라는 뜻
# %Y%m%d 형식으로 된 X를 pandas의 to_datetime 함수를 통해 datetime object로 변환
con_ID1['user_num']=1   #메세지 수를 합산하기 위해 만든 컬럼
con_ID1.set_index(con_ID1['datetime'], inplace=True)  #datetime 컬럼을 index로 만듬 

con_ID1=con_ID1.drop('datetime',1)  #기존에 만들었던 datetime 컬럼을 삭제
df=pd.merge(db1,con_ID1,how='left')
df.head(100) #100개만


#  <a id='the_destination3'></a>
#    
#    # 2.Text Mining_품사별 어휘 분석
#   

# <a id='the_destination4'></a>
# ## 2.1 텍스트 불러오기

# In[55]:

df['text']
dff = df['text']
dff.head(100)


# <a id='the_destination5'></a>
# ## 2.2 명사 

# <a id='the_destination6'></a>
# ### 2.2.1 명사 추출

# In[56]:

brother_tae_change = str(list(df['text'])) ###################형태 변환

import nltk
from konlpy.tag import Twitter
t = Twitter()

noun_comehere = t.nouns(brother_tae_change)   ################명사 추출
noun_comehere


# <a id='the_destination7'></a>
# <b> 한 자리수 이상의 명사 추출 </b>

# In[57]:

noun_comehere1 = [noun_comehere for noun_comehere in noun_comehere if len(noun_comehere) > 1 ]
noun_comehere1


# <a id='the_destination8'></a>
# ### 2.2.2 명사어휘 빈도 및 그래프

# In[58]:

ko = nltk.Text(noun_comehere1)
ko


# <a id='the_destination9'></a>
# #### A. MATPLOTLIB 그래프

# In[59]:

from matplotlib import pylab
from matplotlib import font_manager, rc 
font_fname = 'C:/Anaconda3/Lib/site-packages/pytagcloud/fonts/NanumBarunGothic.ttf' # A font of your choice
font_name = font_manager.FontProperties(fname=font_fname).get_name() 
rc('font', family=font_name)


ko.plot(20)


# <a id='the_destination10'></a>
# #### B. 명사, 빈도수 추출

# In[60]:

ko100 = ko.vocab().most_common(100)
ko100


# <a id='the_destination11'></a>
# #### C. 명사 빈도수 데이터 DB로 변환

# In[61]:

brother_tae_change = str(list(df['text']))

import nltk
from konlpy.tag import Twitter
t = Twitter()
noun_comehere = t.nouns(brother_tae_change) 
noun_comehere
noun_comehere1 = [noun_comehere for noun_comehere in noun_comehere if len(noun_comehere) > 1 ]
noun_comehere1
ko = nltk.Text(noun_comehere1)


ko10=ko.vocab().most_common(10)
db1= DataFrame(ko10)
db1[[0,]]


# 
# <a id='the_destination12'></a>
# #### D. PLOTLY 그래프

# In[63]:

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
#막대그래프o


py.sign_in('simwooinfunnywork','cGhAtiBeOsd3YTlm5xZQ')

data = [
    go.Bar(
        x=db1[0],
        y=db1[1],
        name='Top10',
        marker=dict(
        color='rgb(204,204,204)',
    ))
        ] 
layout = plotly.graph_objs.Layout(
    title='TOP 10 Bar-chart'
)
 
figure = plotly.graph_objs.Figure(
    data=data, layout=layout
)
 

py.iplot(figure, filename='basic_bar_chart.html')


# <a id='the_destination13'></a>
# #### E. 워드 클라우드

# In[64]:

data = ko.vocab().most_common(500)
tmp_data = dict(data)
from wordcloud import WordCloud
wordcloud = WordCloud(font_path='C:/Anaconda3/Lib/site-packages/pytagcloud/fonts/NanumBarunGothic.ttf',
                       relative_scaling = 0.2,
                       background_color='white',
                      ).generate_from_frequencies(tmp_data)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# <a id='the_destination14'></a>
#    ## 2.3 형용사, 동사 

# <a id='the_destination15'></a>
# ###  2.3.1 형용사, 동사 어휘 추출

#  <a id='the_destination16'></a>   
#        
# # 3. Text Mining_연관성 분석

# <a id='the_destination17'></a>
# ## 3.1 KOMORAN

# <a id='the_destination18'></a>
# ### 3.1.1 KOMORAN 형태소 분석

# In[65]:

#2.1에서 dff 데이터 사용 예정
#noun_comehere2 = [noun_comehere for noun_comehere in noun_comehere if len(noun_comehere) > 1 ]
#noun_comehere2

lines = list(dff)  ##################################################  dff 리스트화하고 lines이라 칭함

sentences = [line for line in lines if line != '']   ############### 빈 문장 제거 후 sentences라 칭함


for line in lines[:10]:
    if line != '':
        print(line)

from konlpy.tag import Komoran
tagger = Komoran()
tags = tagger.pos(sentences[0])

tagged_sentences = [tagger.pos(sent) for sent in sentences]

tagged_sentences


# <a id='the_destination19'></a>
# ### 3.1.2 KOMORAN 명사 추출
# 명사 리스트 만들어 보기 / 태그가 NNP, NNG로 시작하는 명사를 리스트

# In[66]:

noun_list = []

for sent in tagged_sentences:    
    for word, tag in sent:
        if tag in ['NNP', 'NNG']:
            noun_list.append(word)
noun_list[:10]


# <a id='the_destination20'></a>
# ### 3.1.3 KOMORAN 명사 빈도수 추출
# collection library를 이용하여 빈도수 계산하기

# In[67]:

from collections import Counter

noun_counts = Counter(noun_list)
noun_counts.most_common(50)


# <a id='the_destination21'></a>
# ### 3.1.4 KOMORAN 시각화
# 결과를 시각화 하기 위한 Matplotlib

# In[71]:

import nltk
import matplotlib.pyplot as plt # 결과를 시각화 하기 위한 matplotlib
get_ipython().magic('matplotlib inline')

from matplotlib import font_manager, rc
path = 'C:/Anaconda3/Lib/site-packages/pytagcloud/fonts/NanumGothic.ttf'     # A font of your choice
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)

# word index 대신 word를 보여주는 그래프
freqdist = nltk.FreqDist(noun_counts)

plt.figure(figsize=(15,3))
freqdist.plot(50)

plt.figure(figsize=(15,3))
freqdist.plot(50,cumulative=True)


# <a id='the_destination22'></a>
# ### 3.1.5 Unique한 명사 리스트 만들기
# 

# In[74]:

unique_nouns = set() #//list
unique_list = []



for sent in tagged_sentences:
    for word, tag in sent:
        if tag in ['NNP','NNG']:
            if word not in unique_list:
                unique_list.append(word)
                
for sent in tagged_sentences:    
    for word, tag in sent:
        if tag in ['NNP', 'NNG']:
            unique_nouns.add(word)

unique_nouns = list(unique_nouns)
noun_index = {noun: i for i, noun in enumerate(unique_nouns)} # 딕셔너리 형태의 자료구조
noun_index


# <a id='the_destination23'></a>
# ### 3.1.6 문장-단어 행렬
# 문장 길이 X 명사 종류 matrix 생성

# In[75]:

import numpy as np
occurs = np.zeros([len(tagged_sentences), len(unique_nouns)])
np.shape(occurs)


# In[76]:

for i, sent in enumerate(tagged_sentences):
    for word, tag in sent:
        if tag in ['NNP', 'NNG']:
            index = noun_index[word]  # 명사가 있으면, 그 명사의 인덱스를 index에 저정
            occurs[i][index] = 1  # 문장 i의 index 자리에 1을 채워 넣는다.
            
occurs[0]


# <a id='the_destination24'></a>
# ### 3.1.7 공존 단어 행렬 계산

# In[77]:

# i 번째 단어
co_occurs = occurs.T.dot(occurs)
co_occurs


# In[78]:

for i in range(100):
    for j in range(100):
        if (co_occurs[i][j] > 1) & (i>j):
            print(unique_nouns[i], unique_nouns[j], co_occurs[i][j])


# <a id='the_destination25'></a>
# ### 3.1.8 네트워크 그래프

# In[79]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


import networkx as nx
graph = nx.Graph()

for i in range(len(unique_nouns)):
    for j in range(i + 1, len(unique_nouns)):
        if co_occurs[i][j] > 20:
            graph.add_edge(unique_nouns[i], unique_nouns[j])
            

krfont = {'family' : 'nanumgothic', 'weight' : 'bold', 'size'   : 10}
plt.rc('font',**krfont)            


plt.figure(figsize=(15, 10))
layout = nx.spring_layout(graph, k=.1)
nx.draw(graph, pos=layout, with_labels=True,
        font_size=20, font_family='AppleGothic',
        alpha=0.3, node_size=3300)
plt.show()


# <a id='the_destination26'></a>
# ## 3.2 TWITTER

# <a id='the_destination27'></a>
# ### 3.2.1 TWITTER 형태소 분석

# In[80]:

#2.1에서 dff 데이터 사용 예정


lines = list(dff)  ##################################################  dff 리스트화하고 lines이라 칭함

sentences = [line for line in lines if line != '']   ############### 빈 문장 제거 후 sentences라 칭함


for line in lines[:3]:
    if line != '':
        print(line)

from konlpy.tag import Twitter
tagger = Twitter()
tags = tagger.pos(sentences[0])

tagged_sentences = [tagger.pos(sent) for sent in sentences]
tagged_sentences


# <a id='the_destination28'></a>
# ### 3.2.2 TWITTER 명사 추출

# In[81]:

# 명사 리스트 만들어 보기 / 태그가 NNP, NNG로 시작하는 명사를 리스트
noun_listt = []

for sent in tagged_sentences:    
    for word, tag in sent:
        if tag in ['Noun']:
            noun_listt.append(word)
noun_listt[:10]



# <a id='the_destination29'></a>
# ### 3.2.3 TWITTER 명사 빈도수 추출

# In[82]:

# collection library를 이용하여 빈도수 계산하기
from collections import Counter

noun_countss = Counter(noun_listt)
noun_countss.most_common(50)


# <a id='the_destination30'></a>
# ### 3.2.4 TWITTER 시각화

# In[83]:

import nltk
import matplotlib.pyplot as plt # 결과를 시각화 하기 위한 matplotlib
get_ipython().magic('matplotlib inline')

# word index 대신 word를 보여주는 그래프
freqdist = nltk.FreqDist(noun_countss)

plt.figure(figsize=(15,3))
freqdist.plot(50)

plt.figure(figsize=(15,3))
freqdist.plot(50,cumulative=True)


# <a id='the_destination31'></a>
# ### 3.2.5 Unique한 명사 리스트 만들기

# In[84]:

# unique한 명사 리스트 만들기

unique_nounss = set()
unique_listt = []

for sent in tagged_sentences:
    for word, tag in sent:
        if tag in ['Noun']:
            if word not in unique_listt:
                unique_listt.append(word)
                
for sent in tagged_sentences:    
    for word, tag in sent:
        if tag in ['Noun']:
            unique_nounss.add(word)

unique_nounss = list(unique_nounss)
noun_index = {noun: i for i, noun in enumerate(unique_nounss)} # 딕셔너리 형태의 자료구조
noun_index


# <a id='the_destination32'></a>
# ### 3.2.6 문자-단어 행렬

# In[85]:


import numpy as np
# 문장 길이 X 명사 종류 matrix 생성
occurss = np.zeros([len(tagged_sentences), len(unique_nounss)])
np.shape(occurss)


# In[86]:

for i, sent in enumerate(tagged_sentences):
    for word, tag in sent:
        if tag in ['Noun']:
            index = noun_index[word]  # 명사가 있으면, 그 명사의 인덱스를 index에 저정
            occurss[i][index] = 1  # 문장 i의 index 자리에 1을 채워 넣는다.
            
occurss[0]


# <a id='the_destination33'></a>
# ### 3.2.7 공존-단어 행렬 계산

# In[87]:

# 공존 단어 행렬 계산

# i 번째 단어
co_occurss = occurss.T.dot(occurss)


# In[88]:

for i in range(100):
    for j in range(100):
        if (co_occurss[i][j] > 1) & (i>j):
            print(unique_nounss[i], unique_nounss[j], co_occurss[i][j])


# <a id='the_destination34'></a>
# ### 3.2.8 네트워크 그래프

# In[89]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


import networkx as nx
graph = nx.Graph()

for i in range(len(unique_nounss)):
    for j in range(i + 1, len(unique_nounss)):
        if co_occurss[i][j] > 24:
            graph.add_edge(unique_nounss[i], unique_nounss[j])
            

krfont = {'family' : 'nanumgothic', 'weight' : 'bold', 'size'   : 10}
plt.rc('font',**krfont)            


plt.figure(figsize=(15, 10))
layout = nx.spring_layout(graph, k=.1)
nx.draw(graph, pos=layout, with_labels=True,
        font_size=20, font_family='AppleGothic',
        alpha=0.3, node_size=3300)
plt.show()


# # 4. Text Mining_감성분석

#    ## 4.1 긍부정 트랜드 출력

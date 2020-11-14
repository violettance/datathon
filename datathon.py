#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="display:table-cell; vertical-align:middle; text-align:center">
#     
# <img src="https://mtstaj.co/datajob/assets/img/logo-hero.png" width="250">
#     
#    
# </div>

# Bir perakende zinciri, mağazalarının içecek taleplerini önceden tahmin ederek stok planlaması yapmak istemektedir. Bu amaçla:
# 
# - Mağazalardan gelecek içecek taleplerinin 3 aylık periyotlarda tahminlerinin elde edilmesi gerekmektedir.
# 
# - Perakende zincirinin geçmiş verileri kullanılarak, makine öğrenmesi ile gerçeklenen bir servisin geliştirilmesi amaçlanmaktadır. Hazırlanacak servis ile gelecek ayların ay bazında talep miktarları önceden belirlenmek istenmektedir.
# 
# - Sizden, aşağıda linkleri verilen verileri kullanarak makine öğrenmesi modelleri geliştirmeniz ve bu modellerin bu amaç doğrultusunda kullanılıp kullanılamayacağını değerlendirmeniz beklenmektedir.
# 
# - Perakende zinciri Amerika Birleşik Devletleri'nde hizmet vermektedir.

# ### Veri Kümeleri

# Çalışmanızın başlangıcında, aşağıda linkleri verilen veri kümelerini bilgisayarınıza indiriniz.
# 
# * Veri dört farklı dosyada sunulmuştur.
# * Bu dosyalardan istediklerinizi kullanabilirsiniz.
# * Eğer birden fazla dosyada bulunan verileri kullanmak istiyorsanız, bu verileri Python ortamında birleştirmelisiniz.
# 
# 
# [1 - Mağaza ve Ürün bazında satış bilgileri](https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/sales.csv)
# 
# [2 - Ürün Bilgileri](https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/items.csv)
# 
# [3 - Mağaza Bilgileri](https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/stores.csv)
# 
# [4 - Ürünlerin Ay Bazında Satış ve Üretim Maliyetleri](https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/costs.csv)

# ### Değerlendirme
# 
# 1. Bu notebookun değerlendirme aşamasında, 
# 
#     * Probleme yaklaşım biçiminiz, 
#     * Kodlama kabiliyetleriniz, 
#     * İstatistik ve görselleştirme gibi teknik bilgileriniz değerlendirilecektir.
# 
# 
# 2. Problem zaman serisi türünden bir veri içermesine rağmen, zaman serisi metodlarına bağlı kalmadan dilediğiniz bir yöntemle probleme yaklaşabilirsiniz.

# ## <font color=darkred>Veri Ön İşleme</font>
# 
# Bu kısımda, size verilen veri kümelerini Python ortamına yüklemeniz ve çalışmanın geri kalanı için veride bulunan problemleri gidermeniz beklenmektedir. Bu amaçla:
# 
# * Verideki problemleri tespit amaçlı görseller çizdirebilirsiniz.
# * Veri türlerini incelemek için kod parçaları yazabilirsiniz.
# * Veriyi sonraki analizlerinizde kolayca işleyebilmek için, veriyi özel veri yapılarında tutabilirsiniz (Pandas DataFrame gibi).

# In[1]:


import pandas as pd
import numpy as np
import requests
from io import StringIO


url1 = "https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/sales.csv"
req = requests.get(url1)
data = StringIO(req.text)

df = pd.read_csv(data)
print(df)

#i got an url error "SSL: CERTIFICATE_VERIFY_FAILED" so i find a way to fix it. the code;
#url1 = "https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/sales.csv"
#df = pd.read_csv(data)
#print(df)


# In[2]:


df['year'].unique()
#last 9 years of data


# In[3]:


print(df.dtypes)


# In[4]:


df.columns


# In[5]:


headers = ["store_number", "item_number", "bottles_sold", "year", "month"]
df.columns = headers
df.head()


# In[6]:


df.tail()


# In[7]:


df.drop(df.loc[:, 'store_number':'item_number'].columns, axis = 1, inplace=True)
df
#not forget to solve year-month spliting later


# In[8]:


df[['bottles_sold']].describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x="year", y="bottles_sold", data=df)
plt.ylim(0,2500)


# In[10]:


df['year'].value_counts()


# In[11]:


missing_data = df.isnull()
missing_data.head(5)


# In[12]:


for column in missing_data.columns.values.tolist():
    print (missing_data[column].value_counts())   


# In[13]:


url2 = "https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/items.csv"
req = requests.get(url2)
data = StringIO(req.text)

df2 = pd.read_csv(data)
print(df2)


# In[14]:


df2.drop(df2.loc[:, 'item_number':'category'].columns, axis = 1, inplace=True)
df2


# In[15]:


df2['bottle_volume_ml'].unique()

#df2['pack'].unique()

#decide important or not later


# In[16]:


for column in missing_data.columns.values.tolist():
    print (missing_data[column].value_counts())


# In[17]:


df2.shape


# In[18]:


df2.dropna(inplace=True)
df2.isnull().sum()

#we already have enough value the volume of the products


# In[19]:


for column in missing_data.columns.values.tolist():
    print (missing_data[column].value_counts())
    
#but obivously not working: look if time remains!


# In[20]:


url3 = "https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/stores.csv"
req = requests.get(url3)
data = StringIO(req.text)

df3 = pd.read_csv(data)
print(df3)


# In[21]:


df3.drop(columns =['store_number', 'county_number'], inplace=True) 
df3

#i dropped two columns because it seems unrelative to analysis.


#i try to reverse coordinates to address, if any time remains in the end: look again!
###from geopy.geocoders import Nominatim
#geolocator = Nominatim(user_agent=",")
#coordinates = (41.238092 , -95.8792)
#location = geolocator.reverse(coordinates)


# In[22]:


print(df3.dtypes)


# In[23]:


for column in missing_data.columns.values.tolist():
    print (missing_data[column].value_counts())


# In[24]:


df3 = df3.reset_index(drop=True)
df3


# In[ ]:


url4 = "https://datatjob-datathon.s3.eu-central-1.amazonaws.com/datajob/costs.csv"
req = requests.get(url4)
data = StringIO(req.text)

df4 = pd.read_csv(data)
print(df4)


# In[ ]:


df4.drop(columns =['item_number'], inplace=True) 
df4


# In[ ]:


df4['state_bottle_retail'].unique()
#no useful value in here.


# ## <font color=darkred>Veri görselleştirme ve Keşifsel Veri Analizi</font>

# Bu kısımda, veriyi tanımanız, keşifsel veri analizi yapmanız ve verinin hikayesini anlatacak görseller çizdirmeniz beklenmektedir. Bu amaçla:
# 
# * Veriyi tanımanıza yardımcı olacak görseller hazırlamalı ve çıkardığınız sonuçları tartışmalısınız. 
# * İstediğiniz grafik tipini kullanabilir ve görselleştirmek için yeni değerler üretebilirsiniz. 
# * Önceki kısımda, verideki problemleri tespit amacıyla hazırladığınız görselleştirmeleri bu bölümde yinelemeyiniz.
# 
# Size yol göstermesi açısından, aşağıda bazı görevler yazılmıştır. Bu görevleri yapmanız beklenmekle beraber, bunlarla sınırlı kalmanıza gerek olmaksızın başka analizler de yapabilirsiniz.

# <font color=darkgreen>Tanımlayıcı İstatistiklerin çıkarılması</font>
# 
#  - Veri setinde bulunan özelliklerin tanımlayıcı istatistiklerini çıkarak yorumlayınız.

# In[ ]:


df.describe()

#year or month irrelevant in this case. Sum of bottles sold is: 27707.000000 in last 9 years. 
#


# In[ ]:


df.mode()

# /*
# we could say most active year of sales is 2016. there may be many reasons for this.
# but most repating number is 5 which is May. it's seems relevant because 
# the weather is getting warmer and people are getting thirsty inherently
# */

#multiple line command is not working


# <font color=darkgreen>Değişkenlerin istatistiksel dağılımlarını inceleyiniz. </font>
# 

# In[ ]:


df2.describe()

#statistical description does not make much sense


# In[ ]:


df3.describe()

#same as df2 but zip_code could be useful some case. maybe colder states get fewer drink
#in that case we could develop hot drink to increase sales and revenue, but it must be 
#examined advanced level


# In[ ]:


df4.describe()

#there is an error. other 3 columns not demonstrated but i check the df4 everyting seems fine


# In[26]:


df4.mode()


# ## <font color=darkred>Verilerin Kaydedilmesi</font>
# 
# Tamizlediğiniz ve yeni özellikler oluşturup modelinizde kullanmaya hazır hale getirdiğiniz veri kümesini csv dosyası olarak kaydediniz. Bu dosyayı, ikinci gün verilerinizi yüklemek için kullanacaksınız.

# In[ ]:


df.to_csv("datathon_irem_kurt_modified.csv", index=True)


# In[ ]:


df = df.append(df4, ignore_index=True)


# ## <font color=darkred>Model Geliştirme</font>

# Bu kısımda gelecek üç ay için içecek talep tahminleri elde edebileceğiniz makine öğrenmesi modelleri geliştirmeniz beklenmektedir.

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("datathon_irem_kurt.csv")
print(df)


# In[28]:


schema_df = pd.read_csv("datathon_irem_kurt.csv")
df.head()

df.shape


# In[29]:


df.state_bottle_cost.fillna(method="bfill", inplace=True)


# In[30]:


df.state_bottle_retail.fillna(method="bfill", inplace=True)


# In[31]:


missing_data = df.isnull()
missing_data

#i filled NaN values with the next row because i think it will be mcuh more efficient;
#than fill mean, mode or median values


# In[32]:


for column in missing_data.columns.values.tolist():
    print (missing_data[column].value_counts())
    
#bottles_sold column has 52170 NaN values in order to fix it we could use mean of each year.


# In[33]:


year_grp = df.groupby(['year'])
year_grp['bottles_sold'].mean()

#i could fill the blanks with mean, maybe it's not the best way but i have to move on :(


# In[34]:


df.fillna(df.mean(), inplace=True)


# In[35]:


missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print (missing_data[column].value_counts())


# In[36]:


df['year'].unique()


# In[37]:


df.month.unique()

#should done it before...


# In[38]:


df.replace({'month' : { 'JAN' : '1', 'FEB' : '2', 'MAR' : '3','APR' : '4', 'MAY' : '5', 'JUN' : '6', 'JUL' : '7', 'AUG' : '8', 'SEP' : '9', 'OCT' : '10', 'NOV' : '1', 'DEC' : '12' }}, inplace=True)


# In[39]:


df.month.unique()


# <font color=darkgreen>Talep tahmin problemini makine öğrenimi metodları kullanarak modelleyiniz. Modelleme aşamasında birden fazla olmak kaydıyla istediğiniz sayıda model kullanabilirsiniz. Seçtiğiniz model / modellerin probleme uygun olup olmadığını tartışınız.</font>

# In[40]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

X = df[['year']]
Y = df['bottles_sold']

lm.fit(X,Y)
Yhat=lm.predict(X)

Yhat[0:12]

#decide to use regression method and linear regressin model for interpretion.
#describe x,y>fit the model>predict


# In[41]:


lm.intercept_

#sold botttle remain almost same each year, we need to look monthly increase/decrease


# In[42]:


df['state_bottle_retail']


# In[43]:


df['state_bottle_retail'] = pd.to_numeric(df['state_bottle_retail'], errors='coerce')
df['state_bottle_retail']

#string to float


# In[44]:


df['state_bottle_cost'] = pd.to_numeric(df['state_bottle_cost'], errors='coerce')
df['state_bottle_cost']

#string to float


# In[45]:


np.isnan(X)

#there are still NaN values, so i decide to drop it :(


# In[46]:


df.dropna()


# In[47]:


X = df[['month']]
Y = df['bottles_sold']

lm.fit(X,Y)
Yhat=lm.predict(X)

Yhat[0:20]


# In[48]:


np.isfinite(df.all())
np.where(df.values == np.finfo(np.float64).max)

#i got an error > Input contains NaN, infinity or a value too large for dtype('float64').
#so i check is finite and also is NaN but i think the problem is "value oo large for dtype"
#done


# In[49]:


df=df.fillna({'a':0}).dropna()


# In[50]:


X = df[['state_bottle_cost']]
Y = df['bottles_sold']

lm.fit(X,Y)
Yhat=lm.predict(X)

Yhat[0:20]


# In[51]:


Z = df[['state_bottle_cost', 'state_bottle_retail',]]
lm.fit(Z, df['bottles_sold'])


# In[52]:


lm.intercept_


# In[53]:


lm.coef_

#both bottle cost andbottle retail increases one unit, impact of increasing is extremely low
#it's seems relevant to me because of the marginal cost definition


# In[54]:


ik = df.groupby('year')
ik.first()


# In[55]:


y_data = df['bottles_sold']
x_data=df.drop('bottles_sold',axis=1)


# In[56]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[57]:


from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lm, x_data[['month']], y_data, cv=4)
Rcross


# In[58]:


print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())


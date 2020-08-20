#Elmas fiyatı hakkında çoklu regresyon modeli kurmamız için verilen  elmas ile ilgili veri setimizde 5 değişken bulunmaktadır.

#Değişkenlerimiz :

#𝑦: Elmasların dolar cinsinden fiyatı (tahmin edilen)
#𝑥1: (CARAT) Elmas karatı: Elmasa özgü ağırlık.
#𝑥2: (DEPTH) Elmasın derinliği: 𝑎, elmas uzunluğu; 𝑏, elmas genişliği ve 𝑐, elmasın derinliği olmak üzere 𝐷𝑒𝑝𝑡ℎ=2𝑐(𝑎+𝑏)⁄ formülünden elde edilmektedir.
#𝑥3: (LENGTH) Elmasın uzunluğu (𝑎).
#𝑥4: (CLARITY) Elmasın temizliği

#Bu değişkenler kullanılarak elmas fiyatının tahmin edilmesi istenmiştir.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

f = open('./data.txt', 'r')
reader = csv.reader(f)
df = pd.read_csv('data.txt',sep=" ")
print(df)

#Veri setimizde görüldüğü üzere 100 satır ve 5 sütun bulunmaktadır.

#Tahmin edeceğimiz bağımlı değişken olan price değişkenini tek bir değişkende tutmak için indexleri kullanarak değişkeni diğer 4 bağımsız değişkenden ayırıyoruz.

price=df.iloc[:, 0].values
price

#Bağımlı değişkene yapmış oldupumuz gibi geri kalan 4 bağımsız değişken içinde indexlerini baz alarak dependents değişkeninde bir araya topluyoruz.

dependents=df.iloc[:,[1,2,3,4]].values
dependents


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
dependents = np.array(ct.fit_transform(dependents))

X_train, X_test, y_train, y_test = train_test_split(dependents, price, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("ROKA_spec.csv", low_memory=False)
df.head()

df1 = df.drop(columns = ['Date', 'Chest', 'arm', 'waist', 'leg', 'head', 'foot']) # 키와 몸무게를 제외한 column 삭제


df1["height"] = pd.to_numeric(df1["height"].str.replace('cm', ''))
df1["weight"] = pd.to_numeric(df1["weight"].str.replace('kg', ''))

df2 = df1.dropna()
# print(df2)

X = df2["height"]
y = df2["weight"]
# plt.plot(X, y, 'o')
# plt.show()

line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1,1), y)

line_fitter.predict([[170]])
line_fitter.coef_

plt.plot(X, y, 'o')
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))
plt.show()


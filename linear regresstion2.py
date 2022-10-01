import pandas as pd

df = pd.read_csv("ROKA_spec.csv", low_memory=False)
df.head()

df1 = df.drop(columns = ['No', 'Date', 'waist'])

df1["Chest"] = pd.to_numeric(df1["Chest"].str.replace('cm', ''))
df1["arm"] = pd.to_numeric(df1["arm"].str.replace('cm', ''))
df1["height"] = pd.to_numeric(df1["height"].str.replace('cm', ''))
df1["leg"] = pd.to_numeric(df1["leg"].str.replace('cm', ''))
df1["head"] = pd.to_numeric(df1["head"].str.replace('cm', ''))
df1["foot"] = pd.to_numeric(df1["foot"].str.replace('cm', ''))
df1["weight"] = pd.to_numeric(df1["weight"].str.replace('kg', ''))

df2 = df1.dropna()
# print(df1)

from sklearn.model_selection import train_test_split
x = df2[['Chest', 'arm', 'height', 'leg', 'head', 'foot']]
y = df2[['weight']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train) 

my_spec = [[95, 95, 178, 86, 54, 28]]
my_predict = mlr.predict(my_spec)

y_predict = mlr.predict(x_test)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Weight")
plt.ylabel("Predicted weight")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

print(mlr.coef_)

# 팔 두께 'arm'과 몸무게 'weight' 의 상관
plt.scatter(df2[['arm']], df2[['weight']], alpha=0.4)
plt.show()

# 키 'height'와 몸무게 'weight'
plt.scatter(df2[['height']], df2[['weight']], alpha=0.4)
plt.show()

# 다리 길이 'leg'과 몸무게 'weight'
plt.scatter(df2[['leg']], df2[['weight']], alpha=0.4)
plt.show()

print(mlr.score(x_train, y_train))
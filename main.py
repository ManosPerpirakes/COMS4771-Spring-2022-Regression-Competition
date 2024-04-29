import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def get_month(var):
    return int(var[0]+var[1])

def get_day(var):
    return int(var[3]+var[4])

def set_data(dataframe):
    dataframe["feature_11"] = dataframe["feature_0"].apply(get_month)
    dataframe["feature_0"] = dataframe["feature_0"].apply(get_day)
    return dataframe

df = pd.read_csv("train_examples.csv")
df = set_data(df)
df_test = pd.read_csv("test_examples.csv")
df_test = set_data(df_test)
X = df
y = pd.read_csv("train_labels.csv")["duration"]
sc = StandardScaler()
X_train = sc.fit_transform(X)
classiefier = KNeighborsRegressor(n_neighbors=300)
classiefier.fit(X, y)
y_pred = classiefier.predict(df_test)
lst = []
for i in range(100000):
    lst.append(i)
result = pd.DataFrame({"id": lst})
result["duration"] = y_pred
result.to_csv("results.csv", index=False)
print("DONE")
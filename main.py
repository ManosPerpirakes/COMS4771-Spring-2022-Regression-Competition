import pandas as pd

df = pd.read_csv("train_examples.csv")
def get_month(var):
    return int(var[0]+var[1])
df["feature_11"] = df["feature_0"].apply(get_month)
def get_time(var):
    return int(var[6]+var[7]+var[9]+var[10]+var[12]+var[13])
df["feature_0"] = df["feature_0"].apply(get_time)
df_test = pd.read_csv("test_examples.csv")
df_test["feature_11"] = df_test["feature_0"].apply(get_month)
df_test["feature_0"] = df_test["feature_0"].apply(get_time)
df_test.to_csv("testfiltered.csv", index=False)
X = df
y = pd.read_csv("train_labels.csv")["duration"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
from sklearn.neighbors import KNeighborsRegressor
classiefier = KNeighborsRegressor(n_neighbors=7)
classiefier.fit(X, y)
y_pred = classiefier.predict(df_test)
lst = []
for i in range(100000):
    lst.append(i)
result = pd.DataFrame({"id": lst})
result["duration"] = y_pred
result.to_csv("results.csv", index=False)
print("DONE")
# Code partially comes from https://www.youtube.com/watch?v=aL21Y-u0SRs
# Financial Management for Young University Students

# This model is unlikely to be ideal

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

d = pd.read_csv("C:/Users/rishi/Documents/AI Innovation Labs 2026/StudentSpending.csv")
df = pd.DataFrame(data=d)

year = ["freshman","sophomore","junior","senior"]
major = ["Engineering","Computer Science", "Economics", "Biology", "Psychology"]
gender = ["female","non-binary","male"]

encYear = OrdinalEncoder(categories = [year])
encMajor = OrdinalEncoder(categories = [major])
encGender = OrdinalEncoder(categories = [gender])

df["year"] = encYear.fit_transform(df[["year"]])
df["major"] = encMajor.fit_transform(df[["major_in_school"]])
df["gender"] = encGender.fit_transform(df[["gender"]])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=11)

from sklearn.linear_model import LogisticRegression

X=df.iloc[0:13]
y=df.iloc[14]

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
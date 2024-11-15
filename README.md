# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Convert emails into numerical feature using tokenization,lowercasing and TF-IDF or Bag of words.
2.Transform the processed text into feature vectors for SVM input.
3. Train an SVM classifier(with a linear or other kernel) on labled data to distinguish between spam and not spam emails.
4. Use the trained SVM model to predict whether new emails are spam and evaluate performance using metrics like accuracy and precision.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: Poovarasu V
RegisterNumber:2305002017  
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()
vectorizer = CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)
model = svm.SVC()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("ACCUARACY:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]
new_message = "Congratulations!"
result = predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```

## Output:

![Screenshot (63)](https://github.com/user-attachments/assets/7dba6c64-63c5-4759-836f-822d1229beaa)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

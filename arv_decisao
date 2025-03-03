import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
recorte_benigno = pd.read_csv('/recorte_benign.csv')
recorte_ransomware = pd.read_csv('/recorte_ransomware.csv')
df = pd.concat([recorte_benigno, recorte_ransomware], axis=0)
for col in df.columns:
    if df[col].dtype == 'object':  
        df[col] = LabelEncoder().fit_transform(df[col])
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
X = df.iloc[:, :-1].values #features
y = df.iloc[:, -1].values #rótulos (benigno ou ransomware)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("acurácia:", accuracy_score(y_test, y_pred))
print("relatório de classificação:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["Benigno", "Ransomware"], yticklabels=["Benigno", "Ransomware"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

from os import pipe
from re import A

import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
df.head()
df["Class"].value_counts(normalize=True)

# Feature Engineering →´processo e criar/transformar variaveis
#                       para melhorar o desempenho do modelo

import numpy as np
df["Amount"] = np.log1p(df["Amount"])

#valores ficam na mesma escala, iependente do tamanho original
from sklearn.preprocessing import StandardScaler 
scaker = StandardScaler()
df["Amount_scaled"] = scaker.fit_transform(df[["Amount"]])

#Separa  a qtd de dados que vão ser usados para treino e teste
from sklearn.model_selection import train_test_split
x = df.drop("Class", axis=1)
y = df["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

#preve se a transação é fraude ou não, é o 1° modelo, serve de referência para os outros
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred= model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Mostra o desempenho do modelo em diferentes limiares de precisão
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
y_probs = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
print("AUC Score:", roc_auc_score(y_test, y_probs))

#mostra o comportamento do modelo com base na precisão e recall
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")    
plt.ylabel("Precision")
plt.show()

#Balanceamento de dados
#Undersampling → reduz a classe majoritaria (1) para ficar do mesmo tamanho da classe minoritaria (0)
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(n=len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])
#Oversampling → cria novos dados baseado nos existentes, aumentando a classe minoritaria (0)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)

# Aprende os cada arvore aprende um padrão dos dados e no final combina as decisões, tende a ser mais preciso
# Arvore de decisão → algoritmo que aprende padrões nos dados e toma decisões com base nesses padrões
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print(classification_report(y_test, y_pred_rf))

#Organiza o fluxo de processamento
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
pipeline.fit(x_train, y_train)
y_pred_pipe = pipeline.predict(x_test)

#Se a probabilidade de fraude for maior que 0.3, então é fraude
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)
print(classification_report(y_test, y_pred_custom))

# XGBoost → algoritmo que combina várias árvores de decisão, cada árvore aprende com os erros da anterior
from xgboost import XGBClassifier
xgb = XGBClassifier(
    scale_pos_weight=10, #ajuda com o desbalanceamento
    use_label_encoder=False,
    eval_metric="logloss"
    )
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
print(classification_report(y_test, y_pred_xgb))
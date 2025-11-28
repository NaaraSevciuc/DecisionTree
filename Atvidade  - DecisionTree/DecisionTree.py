import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv("breast-cancer.csv", na_values='?')
df = df.dropna()

X = pd.get_dummies(df.drop("Class", axis=1))
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)

print("Acurácia do Modelo:", acuracia)
print("\nMatriz de Confusão:\n", matriz)

novos_exemplos = pd.DataFrame([
    {
        "age": "50-59",
        "menopause": "ge40",
        "tumor-size": "20-24",
        "inv-nodes": "0-2",
        "node-caps": "no",
        "deg-malig": 2,
        "breast": "right",
        "breast-quad": "left_up",
        "irradiat": "no"
    }
])

novos_exemplos = pd.get_dummies(novos_exemplos)
novos_exemplos = novos_exemplos.reindex(columns=X.columns, fill_value=0)
pred_novos = model.predict(novos_exemplos)

print("\nClassificação da Nova Instância:", pred_novos[0])

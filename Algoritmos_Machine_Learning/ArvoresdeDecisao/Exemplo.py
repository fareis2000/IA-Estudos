import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import joblib

# === Leitura e pré-processamento ===
df = pd.read_csv(r'c:\Users\farei\OneDrive\Documentos\Fabio\Fabio\Estudos_Python\VC\Partepratica\ArvoresdeDecisao\insurance.csv', keep_default_na=False)
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=[df.columns[7]])
y = df.iloc[:, 7].values

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

X_processed = preprocessor.fit_transform(X)

# === Treinamento ===
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=12)
modelo = DecisionTreeClassifier(max_depth=8, random_state=1, max_leaf_nodes=6)
modelo.fit(X_train, y_train)

# === Função interativa ===
def classificar_usuario():
    print("Responda às seguintes perguntas para classificar a entrada com base no modelo treinado.\n")

    # Coleta das entradas do usuário
    entrada_dict = {}

    for col in categorical_cols:
        opcoes = df[col].unique()
        entrada = input(f"{col} (opções: {', '.join(opcoes)}): ").strip()
        while entrada not in opcoes:
            print(f"Entrada inválida. Digite uma das opções: {', '.join(opcoes)}")
            entrada = input(f"{col}: ").strip()
        entrada_dict[col] = entrada

    for col in numeric_cols:
        while True:
            try:
                entrada = float(input(f"{col} (número): "))
                entrada_dict[col] = entrada
                break
            except ValueError:
                print("Valor inválido. Digite um número.")

    # Cria DataFrame com uma linha
    entrada_df = pd.DataFrame([entrada_dict])

    # Pré-processa a entrada usando o mesmo preprocessor
    entrada_transformada = preprocessor.transform(entrada_df)

    # Predição
    resultado = modelo.predict(entrada_transformada)
    print(f"\n🔍 Resultado da classificação: {resultado[0]}\n")

# === Execução ===
classificar_usuario()

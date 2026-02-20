import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def calcular_metricas(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def validar_modelo(modelo, X, y, cv):
    resultados = {"MAE": [], "MSE": [], "R2": []}

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modelo_fold = clone(modelo)
        modelo_fold.fit(X_train, y_train)
        y_pred = modelo_fold.predict(X_test)

        metricas = calcular_metricas(y_test, y_pred)
        for nome_metrica, valor in metricas.items():
            resultados[nome_metrica].append(valor)

    return resultados


def imprimir_metricas_agregadas(nome_modelo, resultados):
    print(f"\n{nome_modelo} - validação cruzada (5 folds):")
    for metrica, valores in resultados.items():
        media = np.mean(valores)
        desvio = np.std(valores)
        print(f"{metrica}: {media:.2f} ± {desvio:.2f}")


df = pd.read_csv("rides.csv")  # base de dados kaggle
print("Colunas do dataset:", df.columns)
df = df[["Drivers Active Per Hour", "Riders Active Per Hour", "Rides Completed"]]
df = df.dropna(subset=["Rides Completed"])

X = df[["Drivers Active Per Hour", "Riders Active Per Hour"]]
y = df["Rides Completed"]

# Como o dataset não possui coluna temporal explícita, usamos validação cruzada K-Fold.
cv = KFold(n_splits=5, shuffle=True, random_state=42)

modelo_linear = LinearRegression()
modelo_baseline = DummyRegressor(strategy="mean")

resultados_linear = validar_modelo(modelo_linear, X, y, cv)
resultados_baseline = validar_modelo(modelo_baseline, X, y, cv)

print("\nComparação de modelos com validação cruzada")
imprimir_metricas_agregadas("Baseline (média de y_train)", resultados_baseline)
imprimir_metricas_agregadas("Regressão Linear", resultados_linear)

mae_medio_baseline = np.mean(resultados_baseline["MAE"])
mae_medio_linear = np.mean(resultados_linear["MAE"])
if mae_medio_linear < mae_medio_baseline:
    print("\nA regressão linear superou o baseline em MAE médio.")
else:
    print("\nA regressão linear não superou o baseline em MAE médio.")

# Treino final para visualizações.
modelo_linear.fit(X, y)
y_pred_full = modelo_linear.predict(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df["Rides Completed"], bins=30, kde=True, color="blue", ax=axes[0])
axes[0].set_title("Distribuição de Rotas Completas")
axes[0].set_xlabel("Número de Rotas Completas")
axes[0].set_ylabel("Frequência")

axes[1].scatter(y, y_pred_full, alpha=0.7, color="red")
axes[1].plot(
    [min(y), max(y)],
    [min(y), max(y)],
    linestyle="dashed",
    color="black",
)
axes[1].set_xlabel("Valores Reais")
axes[1].set_ylabel("Valores Previstos")
axes[1].set_title("Comparação: Rotas Completas (Reais vs. Previstos)")

residuos = y - y_pred_full
sns.histplot(residuos, bins=30, kde=True, color="purple", ax=axes[2])
axes[2].set_title("Distribuição dos Erros de Previsão (Resíduos)")
axes[2].set_xlabel("Erro")
axes[2].set_ylabel("Frequência")

plt.tight_layout()
plt.show()

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def main():
    # importar o dataframe
    full_df = pickle.load(open("df.sav", "rb"))

    # dividir os dados entre treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(full_df.drop('Attrition', axis=1), full_df["Attrition"],
                                                            test_size=0.30)

    # importar o modelo
    modelo = pickle.load(open("modelo.sav", "rb"))

    modelo.fit(x_treino, y_treino)
    previsoes = modelo.predict(x_teste)
    accuracy = accuracy_score(y_teste, previsoes)

    st.header("Dataframe normalizado")
    st.write(full_df)

    st.header("Previsão")
    st.write(previsoes)

    st.header("Acurácia")
    st.write(accuracy)


if __name__ == "__main__":
    main()

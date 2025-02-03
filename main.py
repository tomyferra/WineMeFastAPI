from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import requests
import os
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Cargar el archivo .env
load_dotenv()
app = FastAPI()

# Obtener las credenciales desde el archivo .env
USER_EMAIL = os.getenv("EMAIL")
USER_PASSWORD = os.getenv("PASSWORD")

# URLs de los endpoints
LOGIN_URL = "https://wineme-api.vercel.app/user/login"
WINES_URL = "https://wineme-api.vercel.app/api/wines"

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes cambiar "*" por el dominio del frontend (ej. "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)



def get_token():
    """Obtener el token de autenticación usando las credenciales del usuario"""
    # Preparar los datos para la solicitud de login
    login_data = {
        "email": USER_EMAIL,
        "password": USER_PASSWORD
    }

    # Hacer la solicitud POST para obtener el token
    response = requests.post(LOGIN_URL, json=login_data)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        # Retornar el token de la respuesta
        return response.json().get("token")
    else:
        raise Exception(f"Error al obtener el token: {response.text}")

def get_all_wines(token):
    """Obtener todos los vinos usando el token de autenticación"""
    # Configurar las cabeceras con el token de autenticación
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Hacer la solicitud GET para obtener todos los vinos
    response = requests.get(WINES_URL, headers=headers)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        # Retornar la lista de vinos
        return response.json()
    else:
        raise Exception(f"Error al obtener los vinos: {response.text}")


def normalizeDF(df):
    ''' normalizes the df '''
    # Convertir las descripciones en una matriz TF-IDF
    vectorizer = TfidfVectorizer()
    df['CombinedTextLong'] = df['Description'] + " " + df['Variety'].fillna("") + " Taninos=" + df['Taninos'].fillna("") + " Madera=" + df['Madera'].fillna("") + " Acidez=" + df['Acidez'].fillna("") + " Cuerpo=" + df['Cuerpo'].fillna("")
    tfidf_matrix_long = vectorizer.fit_transform(df['CombinedTextLong'])
    df['TFIDF_Vector_long'] = list(tfidf_matrix_long.toarray())
    #df = classify_wines(df)
    return tfidf_matrix_long

def similitudCoseno(df):
    # Matriz de TF-IDF (cada fila es un vector TF-IDF de un vino)
    tfidf_matrix_long = np.array(df['TFIDF_Vector_long'].tolist())
    # Calcular la matriz de similitudes
    similarity_matrix_long = cosine_similarity(tfidf_matrix_long)
    return similarity_matrix_long


def getTopSimilarities(similarity_matrix, df, wine_index, top_n=5):
    '''
    Encuentra e imprime las descripciones de los "top_n" vinos más similares al vino especificado por su índice.

    Args:
        similarity_matrix: Matriz de similitudes calculada previamente.
        df: DataFrame con información de los vinos.
        wine_index: Índice del vino para el cual encontrar similitudes.
        top_n: Número de vinos más similares a mostrar (por defecto 5).
    '''
    # Validar que el índice del vino sea válido
    if wine_index < 0 or wine_index >= similarity_matrix.shape[0]:
        print("Índice fuera de rango. Por favor, elige un índice válido.")
        return

    # Obtener las similitudes para el vino elegido
    similarities = similarity_matrix[wine_index]

    # Excluir la similitud consigo mismo estableciendo su valor a -1 temporalmente
    similarities_excl_self = similarities.copy()
    similarities_excl_self[wine_index] = -1

    # Obtener los índices de los "top_n" vinos más similares
    top_similar_indices = similarities_excl_self.argsort()[::-1][:top_n]

    # Imprimir resultados
    current_description = df.iloc[wine_index]['CombinedTextLong']
    current_name = f"{df.iloc[wine_index]['Winery']}, {df.iloc[wine_index]['Name']}"
    print(f"Vino elegido: {current_name} --- Descripción: {current_description}\n")
    df.drop(columns=["TFIDF_Vector_long"], inplace=True)
    df.drop(columns=["__v"], inplace=True)
    print(f"Top {top_n} vinos más similares:")
    results = {"Items": []}
    for idx in top_similar_indices:
        similar_name = f"{df.iloc[idx]['Winery']}, {df.iloc[idx]['Name']}"
        similar_description = df.iloc[idx]['CombinedTextLong']
        similarity_score = similarities[idx]
        results['Items'].append(df.iloc[idx].to_dict())
        print(f"   - {similar_name} --- Descripción: {similar_description}")
    return results

@app.post("/ping")
def ping():
    return {"ping": "pong"}

@app.post("/recommend")
def recommend_wine(wine_input: str):

    # get all wines to recomment:
    token = get_token()
    #get changes
    # Obtener todos los vinos usando el token
    wines = get_all_wines(token)
    df = pd.DataFrame(wines)
    # get the wine id based on the wine_input which is in the Name column of the dataframe
    wine_id = df[df['Name'] == wine_input].index[0]
    tfidf_matrix_long = normalizeDF(df)
    similarity_matrix_long = similitudCoseno(df)
    descriptions = df['CombinedTextLong'].tolist()
    return getTopSimilarities(similarity_matrix_long, df, wine_index = wine_id, top_n=3)

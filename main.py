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
from datetime import datetime
import aiocron
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar el archivo .env
load_dotenv()
app = FastAPI()

# Obtener las credenciales desde el archivo .env
USER_EMAIL = os.getenv("EMAIL")
USER_PASSWORD = os.getenv("PASSWORD")

# URLs de los endpoints
LOGIN_URL = "https://wineme-api-neon.vercel.app/user/login"
WINES_URL = "https://wineme-api-neon.vercel.app/api/wines"

# Health check state
health_status = {
    "is_healthy": True,
    "last_check": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "failures": 0
}

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
    print("Time end: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return results

@app.get("/health")
async def health_check():
    """
    Health check endpoint that returns the current health status
    """
    return {
        "status": "healthy" if health_status["is_healthy"] else "unhealthy",
        "last_check": health_status["last_check"],
        "failures": health_status["failures"]
    }

@app.post("/recommend")
def recommend_wine(wine_input: str):
    print("Time start: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # get all wines to recomment:
    token = get_token()
    # Obtener todos los vinos usando el token
    wines = get_all_wines(token)
    df = pd.DataFrame(wines)
    # get the wine id based on the wine_input which is in the Name column of the dataframe
    wine_id = df[df['Name'] == wine_input].index[0]
    tfidf_matrix_long = normalizeDF(df)
    similarity_matrix_long = similitudCoseno(df)
    descriptions = df['CombinedTextLong'].tolist()
    return getTopSimilarities(similarity_matrix_long, df, wine_index = wine_id, top_n=3)

# Health check function that will be scheduled
async def run_health_check():
    """
    Performs a health check by testing the connection to the API and updating the health status
    """
    global health_status
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Test connection to login endpoint
        login_data = {
            "email": USER_EMAIL,
            "password": USER_PASSWORD
        }
        login_response = requests.post(LOGIN_URL, json=login_data, timeout=10)

        if login_response.status_code != 200:
            raise Exception(f"Login API returned status code: {login_response.status_code}")

        token = login_response.json().get("token")
        if not token:
            raise Exception("No token received from login API")

        # Test connection to wines endpoint
        headers = {"Authorization": f"Bearer {token}"}
        wines_response = requests.get(WINES_URL, headers=headers, timeout=10)

        if wines_response.status_code != 200:
            raise Exception(f"Wines API returned status code: {wines_response.status_code}")

        # If we got here, everything is working properly
        health_status["is_healthy"] = True
        health_status["failures"] = 0
        logger.info("Health check passed successfully")

    except Exception as e:
        health_status["is_healthy"] = False
        health_status["failures"] += 1
        logger.error(f"Health check failed: {str(e)}")

    # Update the last check timestamp regardless of result
    health_status["last_check"] = current_time

# Set up the cron job to run health check every 2 minutes
@aiocron.crontab('*/2 * * * *')  # Run every 2 minutes
async def scheduled_health_check():
    logger.info("Running scheduled health check")
    await run_health_check()

@app.on_event("startup")
async def startup_event():
    # Run an initial health check on startup
    logger.info("Running initial health check on startup")
    await run_health_check()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
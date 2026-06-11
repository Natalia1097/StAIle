import streamlit as st
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from google import genai

st.set_page_config(
    page_title="StAIle",
    page_icon="👔"
)

st.title("👔 StAIle")
st.subheader("IA para Combinação de Cores de Roupas")

# Carregar dataset
with open("colors.json", "r", encoding="utf-8") as f:
    cores_dataset = json.load(f)

st.success(f"Dataset carregado: {len(cores_dataset)} cores")
client = genai.Client(
    api_key=st.secrets["GOOGLE_API_KEY"]
)

# Encontrar cor mais próxima no dataset
def encontrar_cor_mais_proxima(rgb_detectado):

    menor_distancia = float("inf")
    melhor_cor = None

    for cor in cores_dataset:

        rgb_dataset = np.array(cor["rgb"])

        distancia = np.linalg.norm(
            np.array(rgb_detectado) - rgb_dataset
        )

        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_cor = cor

    return melhor_cor


# Extrair cor predominante usando K-Means
def extrair_cor_predominante(img):

    img = img.convert("RGB")

    img = img.resize((200, 200))

    pixels = np.array(img)

    pixels = pixels.reshape((-1, 3))

    kmeans = KMeans(
        n_clusters=3,
        n_init=10,
        random_state=42
    )

    kmeans.fit(pixels)

    labels = kmeans.labels_

    contagem = np.bincount(labels)

    cor_principal = kmeans.cluster_centers_[contagem.argmax()]

    return cor_principal.astype(int)


# Função Gemini
def gerar_recomendacao(nome_cor):

    prompt = f"""
    Você é um consultor de moda profissional.

    A cor principal da roupa é: {nome_cor}

    Gere:

    - Descrição da cor
    - Look casual
    - Look social
    - Cores que combinam
    - Calçados recomendados
    - Acessórios recomendados

    Responda em português e de forma organizada.
    """

    resposta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return resposta.text

if imagem:
        st.write("RGB da Cor Encontrada:")
    st.write(cor_encontrada["rgb"])

    st.subheader("✨ Recomendações StAIle")

    with st.spinner("Gerando sugestões de moda..."):

        recomendacao = gerar_recomendacao(
            cor_encontrada["name"]
        )

    st.markdown(recomendacao)


    

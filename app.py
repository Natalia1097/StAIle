import streamlit as st
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

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


imagem = st.file_uploader(
    "Envie uma foto da roupa"
)

if imagem:

    img = Image.open(imagem)

    st.image(
        img,
        caption="Imagem enviada",
        use_container_width=True
    )

    cor_rgb = extrair_cor_predominante(img)

    cor_encontrada = encontrar_cor_mais_proxima(cor_rgb)

    st.subheader("Cor Predominante Detectada")

    st.success(cor_encontrada["name"])

    st.write("HEX:")
    st.code(cor_encontrada["hex"])

    st.write("RGB Detectado:")
    st.write(cor_rgb.tolist())

    st.write("RGB da Cor Encontrada:")
    st.write(cor_encontrada["rgb"])

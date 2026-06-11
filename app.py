import streamlit as st
import json
import numpy as np
from PIL import Image

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

# Função para encontrar a cor mais próxima
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


imagem = st.file_uploader(
    "Envie uma foto da roupa",
    type=["jpg", "jpeg", "png"]
)

if imagem:

    img = Image.open(imagem)

    st.image(img, caption="Imagem enviada")

    # Reduz tamanho para acelerar
    img = img.resize((100, 100))

    pixels = np.array(img)

    # Média das cores da imagem
    rgb = pixels.mean(axis=(0, 1))

    rgb = rgb.astype(int)

    st.write("RGB detectado:")
    st.write(rgb)

    cor_encontrada = encontrar_cor_mais_proxima(rgb)

    st.subheader("Cor encontrada")

    st.success(cor_encontrada["name"])

    st.write("HEX:")
    st.code(cor_encontrada["hex"])

    st.write("RGB:")
    st.write(cor_encontrada["rgb"])

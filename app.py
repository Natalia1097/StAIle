import streamlit as st
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from google import genai

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="StAIle",
    page_icon="👔",
    layout="wide"
)

st.title("👔 StAIle")
st.subheader("Inteligência Artificial para Combinação de Cores de Roupas")

# =========================
# CARREGAR DATASET
# =========================

try:
    with open("colors.json", "r", encoding="utf-8") as f:
        cores_dataset = json.load(f)

    st.success(f"Dataset carregado: {len(cores_dataset)} cores")

except Exception as e:
    st.error(f"Erro ao carregar colors.json: {e}")
    st.stop()

# =========================
# CONECTAR GEMINI
# =========================

try:
    client = genai.Client(
        api_key=st.secrets["GOOGLE_API_KEY"]
    )

    gemini_ativo = True

except Exception:
    gemini_ativo = False
    st.warning(
        "Gemini não configurado. O reconhecimento de cores continuará funcionando."
    )

# =========================
# FUNÇÃO - COR MAIS PRÓXIMA
# =========================

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

# =========================
# FUNÇÃO - KMEANS
# =========================

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

# =========================
# FUNÇÃO - GEMINI
# =========================

def gerar_recomendacao(nome_cor):

    prompt = f"""
Você é um consultor de moda profissional.

A cor principal da roupa é:

{nome_cor}

Gere:

1. Descrição da cor

2. Look casual

3. Look social

4. Cores que combinam

5. Calçados recomendados

6. Acessórios recomendados

Responda em português.
"""

    resposta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return resposta.text

# =========================
# UPLOAD
# =========================

imagem = st.file_uploader(
    "Envie uma foto da roupa",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESSAMENTO
# =========================

if imagem is not None:

    img = Image.open(imagem)

    st.image(
        img,
        caption="Imagem enviada",
        use_container_width=True
    )

    try:

        cor_rgb = extrair_cor_predominante(img)

        cor_encontrada = encontrar_cor_mais_proxima(cor_rgb)

        st.subheader("🎨 Cor Predominante Detectada")

        col1, col2 = st.columns(2)

        with col1:

            st.success(cor_encontrada["name"])

            st.write("HEX")
            st.code(cor_encontrada["hex"])

        with col2:

            st.write("RGB Detectado")
            st.write(cor_rgb.tolist())

            st.write("RGB da Cor Encontrada")
            st.write(cor_encontrada["rgb"])

        # GEMINI

        if gemini_ativo:

            st.subheader("✨ Recomendações StAIle")

            with st.spinner(
                "Analisando a cor e criando recomendações..."
            ):

                recomendacao = gerar_recomendacao(
                    cor_encontrada["name"]
                )

            st.markdown(recomendacao)

    except Exception as e:

        st.error(f"Erro durante o processamento: {e}")

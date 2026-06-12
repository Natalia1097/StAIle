```python
import streamlit as st
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from google import genai
from datetime import datetime

# =========================
# CONFIGURAÇÃO
# =========================

st.set_page_config(
    page_title="StAIle",
    page_icon="👔",
    layout="wide"
)

st.title("👔 StAIle AI")
st.subheader("Seu assistente inteligente de combinações de roupas")

# =========================
# HISTÓRICO
# =========================

if "historico" not in st.session_state:
    st.session_state.historico = []

# =========================
# CARREGAR DATASET
# =========================

try:
    with open("colors.json", "r", encoding="utf-8") as f:
        cores_dataset = json.load(f)

except Exception as e:
    st.error(f"Erro ao carregar colors.json: {e}")
    st.stop()

# =========================
# GEMINI
# =========================

api_key = st.secrets.get("GOOGLE_API_KEY", None)

if api_key:
    client = genai.Client(api_key=api_key)
    gemini_ativo = True
else:
    gemini_ativo = False
    st.warning("Gemini não configurado.")

# =========================
# CONTEXTO DO USUÁRIO
# =========================

st.sidebar.header("🎯 Informações")

genero = st.sidebar.selectbox(
    "Gênero",
    ["Feminino", "Masculino"]
)

ocasiao = st.sidebar.selectbox(
    "Ocasião",
    ["Casual", "Trabalho", "Festa", "Social"]
)

# =========================
# FUNÇÕES
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


def extrair_cor_predominante(img):
    img = img.convert("RGB")
    img.thumbnail((300, 300))

    pixels = np.array(img).reshape((-1, 3))

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


def gerar_recomendacao(nome_cor, genero, ocasiao):

    prompt = f"""
Você é um consultor de moda.

Cor principal: {nome_cor}
Gênero: {genero}
Ocasião: {ocasiao}

Responda de forma curta e objetiva.

Formato:

🎨 Cor:
(1 frase)

👔 Look:
(1 sugestão completa)

🌈 Combina com:
(apenas 3 cores)

👟 Calçado:
(1 sugestão)

Máximo de 100 palavras.
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
    "Envie uma foto da sua roupa",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESSAMENTO
# =========================

if imagem is not None:

    img = Image.open(imagem)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            img,
            caption="Imagem enviada",
            use_container_width=True
        )

    try:

        cor_rgb = extrair_cor_predominante(img)

        cor_encontrada = encontrar_cor_mais_proxima(cor_rgb)

        with col2:

            st.subheader("🎨 Cor detectada")

            st.success(cor_encontrada["name"])

            st.code(cor_encontrada["hex"])

            st.markdown(
                f"""
                <div
                style="
                width:120px;
                height:120px;
                background:{cor_encontrada['hex']};
                border-radius:12px;
                ">
                </div>
                """,
                unsafe_allow_html=True
            )

        if gemini_ativo:

            st.subheader("✨ Sugestão de Look")

            with st.spinner("Criando sugestão..."):

                recomendacao = gerar_recomendacao(
                    cor_encontrada["name"],
                    genero,
                    ocasiao
                )

            st.markdown(recomendacao)

            st.session_state.historico.append({
                "data": str(datetime.now()),
                "cor": cor_encontrada["name"],
                "genero": genero,
                "ocasiao": ocasiao
            })

    except Exception as e:

        st.error(f"Erro: {e}")

# =========================
# HISTÓRICO
# =========================

st.divider()

st.subheader("📊 Histórico")

if len(st.session_state.historico) == 0:

    st.info("Nenhuma análise realizada.")

else:

    for item in reversed(st.session_state.historico):

        st.write(
            f"👔 {item['cor']} | {item['genero']} | {item['ocasiao']}"
        )

# =========================
# RESET
# =========================

if st.button("🔄 Limpar histórico"):
    st.session_state.historico = []
    st.rerun()
```

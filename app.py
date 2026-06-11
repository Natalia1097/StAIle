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
st.subheader("Seu assistente inteligente de estilo e combinações de roupa")

# =========================
# HISTÓRICO DO USUÁRIO
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
# GEMINI SETUP
# =========================

api_key = st.secrets.get("GOOGLE_API_KEY", None)

if api_key:
    client = genai.Client(api_key=api_key)
    gemini_ativo = True
else:
    gemini_ativo = False
    st.warning("Gemini não configurado. O app funciona apenas com detecção de cor.")

# =========================
# CONTEXTO DO USUÁRIO
# =========================

st.sidebar.header("🎯 Seu estilo")

occasiao = st.sidebar.selectbox(
    "Ocasião",
    ["Casual", "Trabalho", "Festa", "Escola", "Social"]
)

estilo = st.sidebar.selectbox(
    "Estilo pessoal",
    ["Streetwear", "Minimalista", "Elegante", "Criativo", "Esportivo"]
)

# =========================
# FUNÇÕES
# =========================

def encontrar_cor_mais_proxima(rgb_detectado):
    menor_distancia = float("inf")
    melhor_cor = None

    for cor in cores_dataset:
        rgb_dataset = np.array(cor["rgb"])
        distancia = np.linalg.norm(np.array(rgb_detectado) - rgb_dataset)

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


def gerar_recomendacao(nome_cor, ocasiao, estilo):
    prompt = f"""
Você é um consultor de moda premium do StAIle AI.

A cor principal detectada é: {nome_cor}

Contexto do usuário:
- Ocasião: {ocasiao}
- Estilo: {estilo}

Gere um look completo com:

1. Significado da cor
2. Look completo (roupa de cima e baixo)
3. Combinações de cores ideais (até 5)
4. Acessórios recomendados
5. Calçados ideais
6. Erros de estilo a evitar
7. Variação do look

Seja direto, moderno e prático.
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
        st.image(img, caption="Sua imagem", use_container_width=True)

    try:
        # COR
        cor_rgb = extrair_cor_predominante(img)
        cor_encontrada = encontrar_cor_mais_proxima(cor_rgb)

        # UI COR
        with col2:
            st.subheader("🎨 Cor detectada")
            st.success(cor_encontrada["name"])

            st.write("HEX")
            st.code(cor_encontrada["hex"])

            st.write("RGB")
            st.write(cor_rgb.tolist())

            st.markdown(
                f"<div style='width:120px;height:120px;background:{cor_encontrada['hex']};border-radius:10px'></div>",
                unsafe_allow_html=True
            )

        # GEMINI
        if gemini_ativo:
            st.subheader("✨ Seu look inteligente")

            with st.spinner("Criando seu look personalizado..."):
                recomendacao = gerar_recomendacao(
                    cor_encontrada["name"],
                    occasiao,
                    estilo
                )

            st.markdown(recomendacao)

            # SALVAR HISTÓRICO
            st.session_state.historico.append({
                "data": str(datetime.now()),
                "cor": cor_encontrada["name"],
                "occasiao": occasiao,
                "estilo": estilo
            })

        else:
            st.warning("IA desativada (sem API key)")

    except Exception as e:
        st.error(f"Erro: {e}")

# =========================
# HISTÓRICO
# =========================

st.divider()
st.subheader("📊 Seu histórico de estilos")

if len(st.session_state.historico) == 0:
    st.info("Nenhuma análise ainda.")
else:
    for item in reversed(st.session_state.historico):
        st.write(
            f"👔 {item['cor']} | {item['occasiao']} | {item['estilo']} | {item['data']}"
        )

# =========================
# RESET
# =========================

if st.button("🔄 Limpar histórico"):
    st.session_state.historico = []
    st.rerun()

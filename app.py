import streamlit as st
import json

st.set_page_config(
    page_title="StAIle",
    page_icon="👔",
    layout="centered"
)

st.title("👔 StAIle")
st.subheader("IA para Combinação de Cores de Roupas")

with open("colors.json", "r", encoding="utf-8") as f:
    cores_dataset = json.load(f)

st.success(f"Dataset carregado: {len(cores_dataset)} cores")

imagem = st.file_uploader(
    "Envie uma foto da roupa",
    type=["jpg", "jpeg", "png"]
)

if imagem:
    st.image(imagem, caption="Imagem enviada")
    st.info("Próxima etapa: detectar cor da roupa")

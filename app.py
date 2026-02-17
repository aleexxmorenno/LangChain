import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- ESTAS SON LAS LÍNEAS CORREGIDAS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de Estilo
st.set_page_config(page_title="CRITIC-GPT | Sommelier de Ideas", page_icon="🍷", layout="wide")
st.title("🍷 CRITIC-GPT")
st.markdown("### ¿Tu idea es genial o un desastre? Deja que la red lo decida.")

# 2. Sidebar para APIs
with st.sidebar:
    st.header("🔑 Llaves de Acceso")
    google_key = st.text_input("Google API Key:", type="password")
    tavily_key = st.text_input("Tavily API Key:", type="password")
    
    if google_key and tavily_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("Sistemas online. ⚡")

# 3. Lógica con LangChain (Sin LangGraph)
if google_key and tavily_key:
    # Inicializamos herramientas y modelo
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    search_tool = TavilySearchResults(max_results=3)

    # Definimos el Prompt
    prompt = ChatPromptTemplate.from_template("""
    Eres un 'Sommelier de Ideas' con un gusto exquisito y una lengua afilada. 
    Tu trabajo es recibir una idea y compararla con lo que existe en el mundo real.

    CONTEXTO REAL (Búsqueda en la red):
    {context}

    IDEA DEL USUARIO:
    {question}

    INSTRUCCIONES:
    1. Analiza si la idea ya existe o si hay algo parecido en las noticias de hoy.
    2. Da un veredicto 'cool' y sincero: ¿Es un 'Red Flag' 🚩 o es 'Top' 🚀?
    3. Usa un tono sarcástico pero inteligente, muy de 'entrepreneur' de Silicon Valley.
    4. Termina con una recomendación 'pro' para mejorar la idea.
    """)

    # Función para formatear los resultados de búsqueda
    def format_search(query):
        results = search_tool.invoke(query)
        return "\n".join([f"- {r['content']} (Fuente: {r['url']})" for r in results])

    # Construcción de la Chain (LangChain Expression Language - LCEL)
    chain = (
        {"context": format_search, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Interfaz de Usuario
    idea = st.text_input("Suelte tu idea aquí, sin miedo:", placeholder="Ej: Una app para alquilar perros por horas")

    if idea:
        with st.spinner("Consultando con mis contactos en Silicon Valley..."):
            try:
                # Ejecución de la cadena
                respuesta = chain.invoke(idea)
                
                st.markdown("---")
                st.subheader("🧐 El Veredicto:")
                st.write(respuesta)
                
            except Exception as e:
                st.error(f"Error en la Matrix: {e}")

else:
    st.warning("👈 Introduce tus credenciales para desbloquear al crítico.")

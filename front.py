import streamlit as st
from rag_gemini import get_gemini_answer

# Configurar la página
st.set_page_config(page_title="Chatbot RAG - Gemini", page_icon="🤖", layout="centered")

# --- SIDEBAR ---
with st.sidebar:
    st.title('🎓 Chatbot Escolar con Gemini')

    api_key = st.text_input("🔑 API Key de Gemini", type="password", value="AIzaSyACiPAL_N90PsmuPQBSkOhjURgWwu9X9os")

    # Parámetros opcionales (no usados aún, pero puedes conectar con temperatura luego si deseas)
    st.markdown("### Ajustes del modelo (por ahora solo informativos)")
    temperature = st.slider('Temperatura', 0.01, 1.0, 0.5, 0.01)
    top_p = st.slider('Top p', 0.01, 1.0, 0.9, 0.01)
    max_length = st.slider('Longitud máxima', 64, 2048, 512, 32)

    # Botón para limpiar historial
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = [{"role": "assistant", "content": "Hola 👋 ¿En qué puedo ayudarte hoy?"}]

# Inicializa historial si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola 👋 ¿En qué puedo ayudarte hoy?"}]

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando... 🤔"):
            try:
                response = get_gemini_answer(prompt, api_key=api_key, max_tokens=max_length)
                full_response = ''
                placeholder = st.empty()
                for char in response:
                    full_response += char
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

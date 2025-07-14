from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def retrieve_context(query_text):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    return results

def generate_prompt(query, context):
    prompt = f"{query}\n\n"
    for i, ctx in enumerate(context):
        prompt += f"<Context{i+1}>\n{ctx[0].page_content}\n</Context{i+1}>\n\n"
    return prompt

def get_gemini_answer(prompt, api_key, max_tokens=512):
    context = retrieve_context(prompt)
    full_prompt = generate_prompt(prompt, context)

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        max_output_tokens=max_tokens
    )

    response = llm.invoke(full_prompt)
    return response.content

# Ejemplo de uso
if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    pregunta = "¿Qué impacto tiene el anime en la cultura japonesa?"
    respuesta = get_gemini_answer(pregunta, api_key=API_KEY)
    print("Gemini responde:\n", respuesta)

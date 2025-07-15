# Needed libraries
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import sys
import os


# Paths in the project
base_folder = os.path.dirname(os.path.abspath(__file__))
# Path to the Vector_DBs directory
vector_dbs_path = os.path.abspath(os.path.join(base_folder, "../Vector_DBs"))
# Update the Vector_DBs path to import the required file
sys.path.append(vector_dbs_path)

# Import vector_ia_db
from vector_ia_db import get_safe_retriever, init_retriever


# Page setup
st.set_page_config(
    page_title="Chatbot test",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# A quick CSS code
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        color: #1f2937;
    }
    .user-message {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }
    .assistant-message {
        background-color: #f9fafb;
        border-left: 4px solid #10b981;
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    try:
        # Initialize the model
        model = OllamaLLM(model="mistral:7b-instruct-v0.2-q8_0")
        retriever = init_retriever(force_recreate=False)  # Activate retriever
        # We are using the same template as in the RAG.ipynb file
        template = """
You are a helpful, friendly, and knowledgeable AI assistant designed to support future international students of NOVA IMS – a leading school of Information Management and Data Science in Lisbon, Portugal.

Your job is to provide accurate, encouraging, and easy-to-understand answers related to:
- NOVA IMS Master’s and Postgraduate programs,
- Portuguese Student VISA application requirements,
- How to obtain residency after arriving in Portugal,
- Finding housing and understanding living costs in Lisbon,
- Other first steps for settling into life as a new international student in Lisbon.

Use the following retrieved documents to provide accurate and relevant responses. You should not mention document names, document IDs, or file references — focus only on delivering a helpful and human response to the user.

Your answers should be:
- Natural and conversational — avoid sounding like you're quoting a file
- Clear and informative — explain concepts simply and accurately
- Supportive and empathetic — acknowledge that moving abroad is a big and exciting step

Avoid technical language or internal references. Your goal is to make international students feel informed, confident, and supported.

---
If the user’s question is not related to NOVA IMS, studying in Portugal, or moving to Lisbon as a student respond with:
*"I'm here to assist with questions about NOVA IMS, sturdent life in Potugal, and related topics — let me know how I can help!"*

If the user’s question is relevant but cannot be answered from the provided documents, say:
*"That’s an important question! Although I don’t have that information at the moment, I recommend reaching out to NOVA IMS directly or consulting the appropriate service for the most accurate and up-to-date details."*

---
Context:
{context}

user: {question}
Assistant:
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        return model, retriever, chain
    except Exception as e:
        st.error(f"Error initializing the system: {str(e)}")
        return None, None, None


def get_rag_response(question, retriever, chain):
    """Generate the answer using RAG"""
    try:
        # Get context from retriever
        context = retriever.invoke(question)

        # Generate response
        response = chain.invoke({"context": context, "question": question})

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main app


def main():
    st.markdown('<h1 class="main-header">🎓 Chatbot test</h1>',
                unsafe_allow_html=True)

    # Initialize system
    model, retriever, chain = initialize_system()

    if model is None or retriever is None or chain is None:
        st.error("Failed to initialize the system. Please check your configuration.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello there! I'm here to assist with questions about NOVA IMS and student life in Portugal."
        })

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask your question about NOVA IMS or studying in Portugal..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {prompt}</div>',
                    unsafe_allow_html=True)

        # Generate response
        with st.spinner("Thinking..."):
            response = get_rag_response(prompt, retriever, chain)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {response}</div>',
                    unsafe_allow_html=True)

        # Rerun to update the display
        st.rerun()

    # Sidebar with info
    with st.sidebar:
        st.markdown("### About")
        # Infor about the chatbot
        st.info("This chatbot helps international students with questions about NOVA IMS and studying in Portugal.")

        if st.button("Clear Chat"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hello there! I'm here to assist with questions about NOVA IMS and student life in Portugal."
            }]
            st.rerun()
        # Set of recommended quesitons
        st.markdown("### Recomended questions")
        st.info("* What is the average monthly rent in Lisbon?\n\n"
                "* How much does transportation cost in Lisbon?\n\n"
                "* What is the overall cost of living in Lisbon?\n\n"
                "* What steps should I take after arriving in Lisbon?\n\n"
                "* What is the Portuguese Social Security Identification Number (NISS), and why do I need it?\n\n"
                "* How can a foreign citizen obtain a NISS?\n\n"
                "* Does Professor Fernando Bação coordinate any academic programs?\n\n"
                "* Does NOVA IMS provide accommodation options?\n\n"
                "* Can you recommend websites for finding accommodation in Lisbon?\n\n"
                "* I’m interested in learning Portuguese. Does the university offer a Portuguese language course?\n\n"
                "* As a higher education student, how can I obtain a residence permit in Portugal?\n\n"
                "* What documents do I need to apply for a residence permit as a higher education student?\n\n"
                "* How can I prove my financial means to qualify for a residence permit?")


if __name__ == "__main__":
    main()

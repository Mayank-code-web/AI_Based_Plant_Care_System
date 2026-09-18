import streamlit as st
from utils.theme import apply_base_theme

# ---------- LOGIN PROTECTION ---------- #

if not st.session_state.get("logged_in", False):
    st.warning("Please login first")
    st.rerun()

# st.set_page_config is called once, in the router (0_home.py).
apply_base_theme()

import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from utils.env import load_env

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from plantbot.knowledge_base import load_or_build_vectorstore, DB_FAISS_PATH

# Load environment variables (path-independent - see utils/env.py)
load_env()
nest_asyncio.apply()

st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-text'>PlantBot is an AI-powered assistant designed to give instant, general plant-care insights.</p>", unsafe_allow_html=True)

# Ensure async loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Constants
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    # Fallback: not everyone gets the .env file picked up correctly on the first try
    # (wrong folder, saved as .env.txt, or the app wasn't restarted). Let people
    # paste the key directly here as well, so they're never fully blocked.
    if "groq_api_key_runtime" not in st.session_state:
        st.session_state.groq_api_key_runtime = ""

    st.error("🔑 PlantBot needs a Groq API key to work.")
    st.caption(f"Looking for a `.env` file at: `{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')}`")

    with st.expander("Option A: paste your key here for this session (quickest)", expanded=True):
        entered_key = st.text_input(
            "Groq API key",
            type="password",
            value=st.session_state.groq_api_key_runtime,
            placeholder="gsk_...",
        )
        if st.button("Use this key"):
            st.session_state.groq_api_key_runtime = entered_key.strip()
            st.rerun()

    st.markdown("""
**Option B: save it permanently in a `.env` file**
1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
2. In the project's root folder (next to `requirements.txt`), create a file literally named `.env`
   (not `.env.txt` — if you're on Windows and using Notepad, choose "Save as type: All files").
3. Add this one line to it, no quotes:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Stop the app completely and re-run `streamlit run 0_home.py` (environment variables are only
   read when the app starts, so a page refresh isn't enough).

A `.env.example` file with this template is already included in the project.
    """)

    if st.session_state.groq_api_key_runtime:
        GROQ_API_KEY = st.session_state.groq_api_key_runtime
    else:
        st.stop()

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    # Automatically creates vectorstore/db_faiss on first use. This fixes
    # fresh Streamlit Cloud deployments where generated binary files are absent.
    return load_or_build_vectorstore()

try:
    with st.spinner("🌱 Preparing PlantBot knowledge base (first run only)..."):
        vectorstore = load_vectorstore()
except Exception as e:
    st.error("⚠️ PlantBot could not prepare its knowledge base.")
    st.code(str(e))
    st.info(
        "Make sure the project contains the files in `data/plantbot data/` "
        "and that the deployment has internet access for the first-time "
        "embedding-model download."
    )
    st.stop()

def get_prompt_template():
    return PromptTemplate(
        template="""You are PlantBot, a friendly and knowledgeable plant-care assistant.
Use the provided context to answer the user's question as helpfully as possible.

Guidelines:
- Write in a warm, clear, conversational tone — not a dry textbook excerpt.
- If the context has relevant information, structure your answer with short sections where useful,
  e.g. **Likely cause**, **What you can do**, **When to get expert help** — only include sections that
  are actually relevant, don't force all of them every time.
- If the user's message is a greeting, very short, or too vague to answer (e.g. a single word,
  or something like "why" with no prior context), don't just say "I don't know" — instead, warmly
  ask a short clarifying question so the conversation can continue.
- If the context truly doesn't contain relevant information for a genuine plant-care question, say so
  honestly rather than guessing, and suggest a local nursery or extension service.
- Never invent facts that aren't supported by the context.
- This is general plant-care information, not professional horticultural or agronomic advice —
  keep that in mind especially for valuable, rare, or commercial plants.

**Context:**
{context}

**Conversation so far / question:**
{question}

Respond now:
        """,
        input_variables=["context", "question"]
    )

PRIMARY_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-20b"


def load_llm(model_name: str):
    """Create a current Groq chat model.

    Llama 3.3 70B was retired on 2026-08-16, so PlantBot no longer
    depends on that deprecated model. GPT-OSS models are current
    production models on Groq.
    """
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    return ChatGroq(
        temperature=0.4,
        model_name=model_name,
        # Do not pass reasoning_effort here: the installed langchain-groq
        # version validates this field more strictly than the raw Groq SDK.
        # Leaving it unset keeps GPT-OSS compatible and avoids the Pydantic
        # validation error seen in the PlantBot UI.
        max_tokens=1200,
        max_retries=2,
    )


def clean_model_response(text: str) -> str:
    """Remove any accidental reasoning markup before it reaches the UI."""
    import re
    text = text or ""
    # Remove complete <think>...</think> blocks, including multiline blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove orphaned tags that can appear after provider/model changes.
    text = re.sub(r"</?think(?:ing)?>", "", text, flags=re.IGNORECASE)
    return text.strip()

def format_sources(source_documents):
    if not source_documents:
        return ""

    # Deduplicate by filename and show just the clean filename, not the full local path
    seen = []
    for doc in source_documents:
        raw = doc.metadata.get("source", "Unknown Source")
        filename = raw.replace("\\", "/").split("/")[-1]
        if filename not in seen:
            seen.append(filename)

    chips = " &nbsp; ".join(f"📄 {name}" for name in seen)
    return f"<div style='margin-top:10px; font-size:13px; color:#6B7280;'>Sources: {chips}</div>"

def main():
    st.title("💬 PlantBot - AI Plant Care Assistant")
    st.markdown("""
        **Ask any plant-care question, and I'll share general guidance based on my knowledge base!**
        🤖🌿 *Powered by AI & Groq*
    """)

    with st.sidebar:
        st.markdown("""
        ### 🔍 About PlantBot:
        - Uses **Groq-hosted AI models** to answer plant-care queries
        - Retrieves relevant plant-care notes from a knowledge base
        - Provides **fast, general, and contextual responses**
        """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"], unsafe_allow_html=True)

    user_query = st.chat_input("Type your plant-care question...")

    if user_query:
        st.chat_message("user").markdown(f"**You:** {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("🤖 PlantBot is thinking..."):
            try:
                if vectorstore is None:
                    st.error("❌ Error: Vector store failed to load.")
                    return

                retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
                last_error = None
                response = None
                model_used = None

                # Try the primary current production model first, then a smaller
                # current production fallback. This prevents PlantBot from
                # breaking when a provider retires or restricts one model.
                # Try current production models in order. If one is unavailable
                # for this API key/project, the next model is attempted.
                for model_name in (PRIMARY_GROQ_MODEL, FALLBACK_GROQ_MODEL, "llama-3.1-8b-instant"):
                    try:
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=load_llm(model_name),
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': get_prompt_template()}
                        )
                        response = qa_chain.invoke({'query': user_query})
                        model_used = model_name
                        break
                    except Exception as model_error:
                        last_error = model_error

                if response is None:
                    error_text = str(last_error) if last_error else "Unknown Groq error"
                    if "model_not_found" in error_text.lower() or "404" in error_text or "permission" in error_text.lower() or "403" in error_text:
                        st.error(
                            "⚠️ PlantBot could not access an available Groq model with this API key. "
                            "Please check the key/project model permissions, then restart Streamlit."
                        )
                    else:
                        st.error(f"⚠️ PlantBot could not generate a response: {error_text}")
                    return

                result = clean_model_response(response.get("result", "⚠️ No response generated."))
                sources = response.get("source_documents", [])
                if not result:
                    result = "I couldn't generate a clean response. Please try asking the question again."

                formatted_response = f"**PlantBot:** {result}\n\n{format_sources(sources)}"
                st.chat_message("assistant").markdown(formatted_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            except Exception as e:
                st.error(f"⚠️ PlantBot error: {str(e)}")

if __name__ == "__main__":
    main()

import os
import json
import hashlib
from datetime import datetime

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Content Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

DATA_FOLDER = "data"
CHROMA_FOLDER = os.path.join(DATA_FOLDER, "chroma")
HISTORY_FILE = os.path.join(DATA_FOLDER, "history.json")

OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ai_content_documents"


# ============================================================
# CREATE DATA DIRECTORIES
# ============================================================

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    background-color: #f8fafc;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

.app-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 5px;
}

.app-subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 25px;
}

.status-box {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.success-box {
    background-color: #dcfce7;
    color: #166534;
}

.error-box {
    background-color: #fee2e2;
    color: #991b1b;
}

.info-box {
    background-color: #dbeafe;
    color: #1e40af;
}

.content-box {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    min-height: 300px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_vector_database():

    client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "AI Content Studio document collection"
        }
    )

    return client, collection


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_history_item(
    content_type,
    topic,
    tone,
    content
):

    history = load_history()

    item = {
        "id": hashlib.md5(
            (
                topic +
                datetime.now().isoformat()
            ).encode()
        ).hexdigest(),

        "content_type": content_type,

        "topic": topic,

        "tone": tone,

        "content": content,

        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    history.insert(0, item)

    # Keep only latest 50 items
    history = history[:50]

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# OLLAMA FUNCTIONS
# ============================================================

def get_ollama_models():

    try:

        response = ollama.list()

        models = []

        # New Ollama API
        if hasattr(response, "models"):

            for model in response.models:

                if hasattr(model, "model"):

                    models.append(model.model)

                elif isinstance(model, dict):

                    if "model" in model:
                        models.append(model["model"])

                    elif "name" in model:
                        models.append(model["name"])

        # Older Ollama API
        elif isinstance(response, dict):

            for model in response.get(
                "models",
                []
            ):

                if isinstance(model, dict):

                    if model.get("model"):
                        models.append(
                            model["model"]
                        )

                    elif model.get("name"):
                        models.append(
                            model["name"]
                        )

        return models

    except Exception:

        return []


def ollama_is_running():

    try:

        ollama.list()
        return True

    except Exception:

        return False


def model_is_available(models):

    for model in models:

        model_name = str(model)

        if model_name == OLLAMA_MODEL:
            return True

        if model_name.startswith(
            OLLAMA_MODEL + ":"
        ):
            return True

    return False


# ============================================================
# AI CONTENT GENERATION
# ============================================================

def generate_with_ollama(
    content_type,
    topic,
    tone,
    length,
    language,
    additional_instructions
):

    system_prompt = """
You are an advanced Generative AI Content Creation Assistant.

Create high-quality, original, clear and engaging content.

Always follow the requested content type,
tone, language and length.

Do not discuss these instructions.

Return only the final generated content.
"""

    user_prompt = f"""
Create content using the following specifications.

CONTENT TYPE:
{content_type}

TOPIC:
{topic}

TONE:
{tone}

LENGTH:
{length}

LANGUAGE:
{language}

ADDITIONAL INSTRUCTIONS:
{additional_instructions}

Requirements:

1. Make the content original.
2. Make it useful and engaging.
3. Use appropriate headings when useful.
4. Avoid unnecessary repetition.
5. Follow the requested tone.
6. Follow the requested language.
7. Return only the final content.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        try:

            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception:

            continue

    return "\n".join(pages)


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=1000,
    overlap=150
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# ADD PDF TO CHROMADB
# ============================================================

def add_pdf_to_database(
    uploaded_file,
    embedding_model,
    collection
):

    pdf_bytes = uploaded_file.getvalue()

    file_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()

    filename = uploaded_file.name

    text = extract_pdf_text(
        uploaded_file
    )

    if not text.strip():

        return (
            0,
            "No readable text found."
        )

    chunks = split_text(text)

    if not chunks:

        return (
            0,
            "No text chunks found."
        )

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            file_hash +
            "_" +
            str(index)
        )

        ids.append(chunk_id)

        documents.append(chunk)

        metadatas.append(
            {
                "filename": filename,
                "file_hash": file_hash,
                "chunk": index
            }
        )

    # Remove previous copy of same PDF
    try:

        collection.delete(
            where={
                "file_hash": file_hash
            }
        )

    except Exception:

        pass

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return (
        len(chunks),
        f"{filename} added successfully."
    )


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    embedding_model,
    collection,
    number_of_results=5
):

    total = collection.count()

    if total == 0:
        return []

    question_embedding = (
        embedding_model
        .encode([question])
        .tolist()[0]
    )

    result = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=min(
            number_of_results,
            total
        )
    )

    documents = result.get(
        "documents",
        [[]]
    )

    if not documents:
        return []

    return documents[0]


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def answer_from_documents(
    question,
    embedding_model,
    collection
):

    documents = search_documents(
        question,
        embedding_model,
        collection
    )

    if not documents:

        return (
            "No relevant documents were found. "
            "Please upload a PDF first."
        )

    context = "\n\n".join(
        documents
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY
the provided document context.

If the answer cannot be found in the context,
say that the information is not available
in the uploaded documents.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Give a clear and concise answer.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# SESSION STATE
# ============================================================

if "generated_content" not in st.session_state:
    st.session_state.generated_content = ""

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""


# ============================================================
# GET OLLAMA STATUS
# ============================================================

available_models = get_ollama_models()

ollama_running = ollama_is_running()

model_available = model_is_available(
    available_models
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ AI Settings")

    st.divider()

    st.write("**AI Model**")

    st.code(
        OLLAMA_MODEL
    )

    st.write("**Embedding Model**")

    st.code(
        EMBEDDING_MODEL
    )

    st.divider()

    if ollama_running:

        st.success(
            "🟢 Ollama is running"
        )

    else:

        st.error(
            "🔴 Ollama is not running"
        )

    if model_available:

        st.success(
            f"🟢 {OLLAMA_MODEL} is installed"
        )

    else:

        st.warning(
            f"🟡 {OLLAMA_MODEL} not found"
        )

    st.divider()

    st.info(
        """
The browser communicates with
Streamlit.

Streamlit communicates with
your local Ollama server.
"""
    )

    st.markdown(
        """
### Platform Features

🤖 AI Generation

📝 Multiple Content Types

🎨 Multiple Writing Tones

🌐 Multiple Languages

📄 PDF Knowledge Base

🧠 RAG Question Answering

🔎 Semantic Search

💾 Content History

⬇️ Download Content
"""
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🤖 AI Content Studio"
)

st.caption(
    "Full-Stack Generative AI Content Creation Platform"
)


# ============================================================
# OLLAMA STATUS
# ============================================================

if not ollama_running:

    st.warning(
        """
⚠️ Ollama is not running.

Please start Ollama on your computer.

Then refresh this browser page.
"""
    )

elif not model_available:

    st.warning(
        f"""
⚠️ Ollama is running, but `{OLLAMA_MODEL}`
was not found.

Open your terminal and run:

ollama pull {OLLAMA_MODEL}
"""
    )

else:

    st.success(
        f"✅ Ollama connected successfully — using {OLLAMA_MODEL}"
    )


# ============================================================
# DASHBOARD
# ============================================================

history = load_history()

try:

    _, collection = load_vector_database()

    document_count = collection.count()

except Exception:

    collection = None
    document_count = 0


st.subheader("📊 Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Generated Contents",
        len(history)
    )

with col2:

    st.metric(
        "Document Chunks",
        document_count
    )

with col3:

    st.metric(
        "Content Types",
        7
    )

with col4:

    st.metric(
        "Writing Tones",
        7
    )


st.divider()


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "✨ AI Content Generator",
        "📚 PDF Knowledge Base",
        "🕘 Content History"
    ]
)


# ============================================================
# TAB 1 - AI CONTENT GENERATOR
# ============================================================

with tab1:

    st.header(
        "✨ Create AI Content"
    )

    left, right = st.columns(
        [1, 1.5]
    )


    # ========================================================
    # INPUT PANEL
    # ========================================================

    with left:

        st.subheader(
            "Content Settings"
        )

        content_type = st.selectbox(
            "Content Type",
            [
                "Blog Article",
                "Social Media Post",
                "Email",
                "Product Description",
                "Advertisement",
                "Video Script",
                "Website Content"
            ]
        )

        topic = st.text_area(
            "Topic",
            placeholder=(
                "Example: Benefits of "
                "Artificial Intelligence in Education"
            ),
            height=150
        )

        tone = st.selectbox(
            "Writing Tone",
            [
                "Professional",
                "Friendly",
                "Creative",
                "Funny",
                "Persuasive",
                "Educational",
                "Formal"
            ]
        )

        length = st.selectbox(
            "Content Length",
            [
                "Short",
                "Medium",
                "Long"
            ]
        )

        language = st.selectbox(
            "Language",
            [
                "English",
                "Hindi",
                "Telugu",
                "Tamil",
                "Kannada",
                "Malayalam"
            ]
        )

        additional_instructions = st.text_area(
            "Additional Instructions",
            placeholder=(
                "Example: Include 5 headings "
                "and a conclusion."
            ),
            height=100
        )

        generate_button = st.button(
            "🚀 Generate AI Content",
            type="primary",
            use_container_width=True
        )


        # ====================================================
        # GENERATE CONTENT
        # ====================================================

        if generate_button:

            if not topic.strip():

                st.error(
                    "Please enter a topic."
                )

            elif not ollama_running:

                st.error(
                    "Ollama is not running. "
                    "Please start Ollama."
                )

            elif not model_available:

                st.error(
                    f"The model '{OLLAMA_MODEL}' "
                    "is not installed."
                )

            else:

                with st.spinner(
                    "🤖 AI is generating your content..."
                ):

                    try:

                        generated = (
                            generate_with_ollama(
                                content_type,
                                topic,
                                tone,
                                length,
                                language,
                                additional_instructions
                            )
                        )

                        st.session_state.generated_content = (
                            generated
                        )

                        st.session_state.last_topic = (
                            topic
                        )

                        save_history_item(
                            content_type,
                            topic,
                            tone,
                            generated
                        )

                        st.success(
                            "✅ Content generated successfully!"
                        )

                    except Exception as error:

                        st.error(
                            "❌ AI generation failed."
                        )

                        st.code(
                            str(error)
                        )


    # ========================================================
    # OUTPUT PANEL
    # ========================================================

    with right:

        st.subheader(
            "🤖 Generated Content"
        )

        if st.session_state.generated_content:

            # Use Streamlit text instead of raw HTML.
            # This prevents HTML tags appearing in the browser.

            st.text_area(
                "Generated Result",
                value=st.session_state.generated_content,
                height=500,
                label_visibility="collapsed"
            )

            st.write("")

            button1, button2 = st.columns(2)

            with button1:

                st.download_button(
                    label="⬇️ Download",
                    data=(
                        st.session_state
                        .generated_content
                    ),
                    file_name=(
                        "ai_generated_content.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True
                )

            with button2:

                if st.button(
                    "🔄 Clear",
                    use_container_width=True
                ):

                    st.session_state.generated_content = ""

                    st.rerun()

        else:

            st.info(
                "Your AI-generated content "
                "will appear here."
            )


# ============================================================
# TAB 2 - PDF KNOWLEDGE BASE
# ============================================================

with tab2:

    st.header(
        "📚 PDF Knowledge Base"
    )

    st.write(
        "Upload PDF documents to create a "
        "searchable AI knowledge base."
    )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    try:

        with st.spinner(
            "Loading embedding model..."
        ):

            embedding_model = (
                load_embedding_model()
            )

        _, collection = (
            load_vector_database()
        )

        embedding_ready = True

    except Exception as error:

        embedding_ready = False

        embedding_model = None

        collection = None

        st.error(
            "❌ Embedding model could not be loaded."
        )

        st.code(
            str(error)
        )


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )


    if uploaded_files:

        st.write(
            f"📄 {len(uploaded_files)} PDF file(s) selected."
        )


    if (
        uploaded_files
        and
        embedding_ready
    ):

        process_button = st.button(
            "📥 Process PDF Documents",
            type="primary"
        )

        if process_button:

            total_chunks = 0

            progress = st.progress(0)

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                try:

                    chunks, message = (
                        add_pdf_to_database(
                            uploaded_file,
                            embedding_model,
                            collection
                        )
                    )

                    total_chunks += chunks

                    st.success(
                        message +
                        f" ({chunks} chunks)"
                    )

                except Exception as error:

                    st.error(
                        f"Error processing "
                        f"{uploaded_file.name}: "
                        f"{error}"
                    )

                progress.progress(
                    (index + 1) /
                    len(uploaded_files)
                )

            st.success(
                f"✅ Processing completed. "
                f"{total_chunks} chunks added."
            )


    st.divider()


    # ========================================================
    # RAG QUESTION ANSWERING
    # ========================================================

    st.subheader(
        "🔎 Ask Questions About Your PDFs"
    )

    question = st.text_area(
        "Your Question",
        placeholder=(
            "Example: What are the main "
            "advantages discussed in the document?"
        ),
        height=100
    )

    ask_button = st.button(
        "🧠 Ask AI",
        type="primary"
    )


    if ask_button:

        if not question.strip():

            st.error(
                "Please enter a question."
            )

        elif not embedding_ready:

            st.error(
                "Embedding model is not available."
            )

        elif collection is None:

            st.error(
                "Vector database is not available."
            )

        elif collection.count() == 0:

            st.warning(
                "Please upload and process "
                "a PDF first."
            )

        elif not ollama_running:

            st.error(
                "Ollama is not running."
            )

        elif not model_available:

            st.error(
                f"{OLLAMA_MODEL} is not installed."
            )

        else:

            with st.spinner(
                "🔎 Searching documents and generating answer..."
            ):

                try:

                    answer = (
                        answer_from_documents(
                            question,
                            embedding_model,
                            collection
                        )
                    )

                    st.subheader(
                        "🤖 AI Answer"
                    )

                    st.write(
                        answer
                    )

                except Exception as error:

                    st.error(
                        "❌ Question answering failed."
                    )

                    st.code(
                        str(error)
                    )


    st.divider()


    if collection is not None:

        st.write(
            "📊 Total indexed document chunks: "
            f"**{collection.count()}**"
        )


# ============================================================
# TAB 3 - CONTENT HISTORY
# ============================================================

with tab3:

    st.header(
        "🕘 Generated Content History"
    )

    history = load_history()

    if not history:

        st.info(
            "No generated content yet."
        )

    else:

        st.write(
            f"Total saved items: **{len(history)}**"
        )

        for item in history:

            title = (
                f"📝 {item['content_type']} — "
                f"{item['topic']}"
            )

            with st.expander(title):

                st.write(
                    f"**Tone:** {item['tone']}"
                )

                st.write(
                    f"**Created:** "
                    f"{item['created_at']}"
                )

                st.divider()

                st.text_area(
                    "Content",
                    value=item["content"],
                    height=300,
                    key=f"text_{item['id']}"
                )

                st.download_button(
                    "⬇️ Download",
                    data=item["content"],
                    file_name=(
                        "content_"
                        +
                        item["id"][:8]
                        +
                        ".txt"
                    ),
                    mime="text/plain",
                    key=(
                        "download_"
                        +
                        item["id"]
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Content Studio | "
    "Python • Streamlit • Ollama • "
    "Sentence Transformers • ChromaDB • PyPDF"
)
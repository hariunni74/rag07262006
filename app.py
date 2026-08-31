import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing from the .env file.")
    st.stop()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


llm = ChatOpenAI(
    model="gpt-5-mini",
    model_kwargs={"seed": 42}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Resume and Job Description Assistant.
Follow these rules:
1. Answer only the specific question asked by the user.
2. Use only the provided resume and job-description context.
3. Give a fit score only when the user specifically asks for:
   - a fit score
   - resume match percentage
   - candidate suitability
   - overall fit analysis
4. For all other questions, provide only the requested information.
5. Do not invent skills, experience, qualifications or figures.
6. If the information is unavailable, say:
   "I don't see this information in the documents."

When a fit score is requested, compute it using this exact rubric.
Do not deviate from these categories or weights, and do not use your own
judgment to invent a different scoring scheme:

- Required skills match: 0-40 points
  (award points proportionally to how many required/must-have skills in the
  job description are demonstrably present in the resume)
- Experience / years relevance: 0-25 points
  (compare years and relevance of prior roles to the job description)
- Education / certifications match: 0-15 points
- Domain / industry relevance: 0-10 points
- Soft skills / extras (leadership, tools, nice-to-haves): 0-10 points

Show each sub-score explicitly, then sum them for the total (out of 100).
This total is the Fit Score. After the rubric breakdown, provide:
- Fit score: X/100
- Verdict (Strong Fit / Moderate Fit / Weak Fit)
- Matching skills
- Missing skills or gaps
- Final recommendation

Context:
{context}"""
    ),
    ("human", "{question}")
])

if "retriever" not in st.session_state:
    st.session_state.retriever = None


def load_pdf(uploaded_file, document_type):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    try:
        documents = PyPDFLoader(temp_path).load()
        for document in documents:
            document.metadata["document_type"] = document_type
        return documents
    finally:
        os.remove(temp_path)


def index_documents(resume_file, jd_file):
    documents = (
        load_pdf(resume_file, "RESUME")
        + load_pdf(jd_file, "JOB DESCRIPTION")
    )
    chunks = splitter.split_documents(documents)

    # Resume + JD are short documents. Retrieve every chunk instead of doing
    # approximate top-k similarity search, so the context passed to the LLM
    # is identical on every question instead of varying with retrieval order.
    st.session_state.retriever = FAISS.from_documents(
        chunks,
        embeddings
    ).as_retriever(
        search_kwargs={"k": max(len(chunks), 1)}
    )
    return len(chunks)


def ask_question(question):
    documents = st.session_state.retriever.invoke(question)
    context = "\n\n".join(
        f"[{document.metadata['document_type']}]\n"
        f"{document.page_content}"
        for document in documents
    )
    messages = prompt.format_messages(
        context=context,
        question=question
    )
    return llm.invoke(messages).content


st.set_page_config(
    page_title="Resume and Job Description Assistant",
    layout="wide"
)
st.title("Resume and Job Description Assistant")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )
with col2:
    jd_file = st.file_uploader(
        "Upload Job Description PDF",
        type=["pdf"]
    )

if st.button("Index Documents"):
    if not resume_file or not jd_file:
        st.warning("Please upload both PDF files.")
    else:
        with st.spinner("Indexing documents..."):
            chunk_count = index_documents(
                resume_file,
                jd_file
            )
        st.success(
            f"Documents indexed successfully: {chunk_count} chunks."
        )

question = st.text_input(
    "Ask a question",
    placeholder="What certifications does the candidate have?"
)

if st.button("Ask"):
    if st.session_state.retriever is None:
        st.warning("Please index the documents first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing..."):
            answer = ask_question(question)
        st.subheader("Answer")
        st.write(answer)
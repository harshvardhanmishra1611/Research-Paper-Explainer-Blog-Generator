import base64
import os
import tempfile

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file_handle:
            encoded_string = base64.b64encode(image_file_handle.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] > .main {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .header, .summary-section {{
                background-color: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(5px);
            }}
            .header h1, .header p {{
                 color: #1E293B;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Background image file not found. Make sure 'background.png' is in the same directory.")

st.set_page_config(
    page_title="PAPER DECODED",
    layout="centered",
    page_icon="📘"
)

add_bg_from_local('assets/background.png')

st.markdown("""
    <style>
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3a7bc8;
            transform: scale(1.02);
        }
        .stFileUploader>div>div>div>button {
            background-color: #4A90E2;
            color: white;
        }
        .summary-section {
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .header {
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .processing {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .success-box {
            background-color: #e6f7ee;
            color: #28a745;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 5px solid #28a745;
        }
        .error-box {
            background-color: #fdecea;
            color: #dc3545;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 5px solid #dc3545;
        }
        .subheader-icon {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("""
        <div class="header">
            <h1 style="text-align: center; margin-bottom: 0.5rem;">📘 PAPER DECODED</h1>
            <p style="text-align: center; font-size: 1.1rem; opacity: 0.9;">
                An intelligent platform that simplifies academic research and transforms it into clear, accessible blog- style content for everyone.
            </p>
        </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## About")
    st.markdown("""
        **PAPER DECODED** helps you quickly understand research papers by:
        - Extracting key insights
        - Simplifying complex concepts
        - Generating blog-style summaries
    """)

    st.markdown("## How to use")
    st.markdown("""
        1. Upload a PDF research paper
        2. Wait for processing to complete
        3. Click "Generate Blog Summary"
        4. Explore the simplified sections
    """)

    st.markdown("## Powered by")
    st.markdown("""
        - [Groq](https://groq.com/) - Ultra-fast LLM
        - [LangChain](https://langchain.com/) - AI orchestration
        - [Streamlit](https://streamlit.io/) - Beautiful apps
    """)

st.markdown("### Upload your research paper")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.status("Processing document...", expanded=True) as status:
        st.markdown("""
            <div class="processing">
                <span class="loader"></span>
                <span>Extracting text from PDF...</span>
            </div>
        """, unsafe_allow_html=True)

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        st.markdown("""
            <div class="processing">
                <span class="loader"></span>
                <span>Splitting document into chunks...</span>
            </div>
        """, unsafe_allow_html=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        st.markdown("""
            <div class="processing">
                <span class="loader"></span>
                <span>Creating vector database...</span>
            </div>
        """, unsafe_allow_html=True)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embedding)

        st.markdown("""
            <div class="processing">
                <span class="loader"></span>
                <span>Initializing AI model...</span>
            </div>
        """, unsafe_allow_html=True)
        llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        status.update(label="Processing complete!", state="complete", expanded=False)

    st.markdown("""
        <div class="success-box">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-check-circle"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span style="font-weight: 500;">Paper processed successfully!</span>
            </div>
            <p style="margin-top: 0.5rem; margin-bottom: 0;">Click the button below to generate your summary.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Generate Blog Summary", use_container_width=True):
            try:
                with st.spinner("Generating insights... This may take a moment"):
                    with st.container():
                        st.markdown("""
                            <div class="summary-section">
                                <div class="subheader-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4A90E2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-alert-circle"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                                    <h3 style="color: #4A90E2; margin: 0;">Problem Statement</h3>
                                </div>
                                <div style="margin-top: 1rem;">
                        """, unsafe_allow_html=True)
                        problem = qa.run(
                            "What problem does this paper solve? Explain simply in 2-3 paragraphs suitable for a blog post.")
                        st.write(problem)
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    with st.container():
                        st.markdown("""
                            <div class="summary-section">
                                <div class="subheader-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4A90E2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>
                                    <h3 style="color: #4A90E2; margin: 0;">Methodology</h3>
                                </div>
                                <div style="margin-top: 1rem;">
                        """, unsafe_allow_html=True)
                        methodology = qa.run(
                            "Explain the methodology in simple terms suitable for a non-expert audience. Use 3-4 paragraphs with clear explanations.")
                        st.write(methodology)
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    with st.container():
                        st.markdown("""
                            <div class="summary-section">
                                <div class="subheader-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4A90E2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-key"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path></svg>
                                    <h3 style="color: #4A90E2; margin: 0;">Key Takeaways</h3>
                                </div>
                                <div style="margin-top: 1rem;">
                        """, unsafe_allow_html=True)
                        takeaways = qa.run(
                            "Summarize the results in 5 bullet points with clear explanations for each point. Format as markdown bullets.")
                        st.markdown(takeaways)
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    with st.container():
                        st.markdown("""
                            <div class="summary-section">
                                <div class="subheader-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4A90E2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bookmark"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path></svg>
                                    <h3 style="color: #4A90E2; margin: 0;">Conclusion</h3>
                                </div>
                                <div style="margin-top: 1rem;">
                        """, unsafe_allow_html=True)
                        conclusion = qa.run(
                            "Summarize the conclusion in a blog-style format with 2-3 paragraphs. Include potential implications and future directions if mentioned in the paper.")
                        st.write(conclusion)
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    st.balloons()

            except Exception as e:
                st.markdown(f"""
                    <div class="error-box">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-alert-triangle"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                            <span style="font-weight: 500;">Something went wrong</span>
                        </div>
                        <p style="margin-top: 0.5rem; margin-bottom: 0;">{str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)

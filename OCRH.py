import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import os
import tempfile
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import io
import platform

# Load environment variables
load_dotenv()

# Configure Tesseract for Windows
if platform.system() == "Windows":
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        st.warning("Tesseract OCR not found at default path. OCR functionality will be limited.")

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    GROQ_MODEL = "llama3-70b-8192"
    TEXT_TOP_K = 5
    MAX_TOKENS = 500
    MIN_TEXT_PER_PAGE = 50  # Threshold to trigger OCR
    OCR_DPI = 200  # Resolution for OCR processing
    MEMORY_LIMIT = 5  # Number of previous exchanges to remember

@st.cache_resource
def load_models():
    """Load ML models and clients"""
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        ),
        "groq_client": Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    }

class DocumentProcessor:
    def __init__(self):
        self.models = load_models()
        
    def process_pdf(self, file_bytes):
        """Process PDF document with fallback to OCR when needed"""
        with st.spinner("Analyzing document..."):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texts = self._extract_text(doc)
            
            if self._needs_ocr(texts, len(doc)):
                st.info("Scanned document detected. Running OCR...")
                texts = self._run_ocr(doc)
            
            if not texts:
                st.error("No readable text found in document")
                return None
                
            text_embeddings = self.models["text_embedder"].encode(
                [t["content"] for t in texts], show_progress_bar=False
            )
                
        return {
            "texts": texts,
            "text_embeddings": np.array(text_embeddings),
            "max_page": max(t["page"] for t in texts)
        }
    
    def _needs_ocr(self, texts, page_count):
        """Determine if OCR is needed based on extracted text"""
        if not texts:
            return True
            
        total_chars = sum(len(t["content"]) for t in texts)
        avg_chars_per_page = total_chars / page_count
        return avg_chars_per_page < Config.MIN_TEXT_PER_PAGE
    
    def _run_ocr(self, doc):
        """Perform OCR on image-based PDF pages"""
        texts = []
        for page_num in range(len(doc)):
            try:
                pix = doc[page_num].get_pixmap(dpi=Config.OCR_DPI)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                
                if text.strip():
                    chunks = self.models["text_splitter"].split_text(text)
                    for chunk in chunks:
                        texts.append({
                            "content": chunk,
                            "page": page_num + 1,
                            "type": "ocr_text"
                        })
            except Exception as e:
                st.warning(f"OCR failed on page {page_num+1}: {str(e)}")
        
        return texts
    
    def _extract_text(self, doc):
        """Extract text from text-based PDF"""
        texts = []
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            if page_text.strip():
                chunks = self.models["text_splitter"].split_text(page_text)
                for chunk in chunks:
                    texts.append({
                        "content": chunk,
                        "page": page_num + 1,
                        "type": "text"
                    })
        return texts

def generate_response(prompt, context, client, conversation_history):
    """Generate LLM response with document context and conversation history"""
    try:
        system_prompt =   """You are a PDF Chat Bot. Provide:
1. Accurate information from the document
2. Well-structured responses with clear sections
3. Do not mention page numbers in the response
4. If document doesn't have the info, say 'This document does not contain information related to your query and give the answer in 2 lines only'
5. Maintain context from previous questions when appropriate"""
        
        
        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious Conversation:\n" + "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in conversation_history
            )
        
        # Combine document context and conversation history
        full_context = f"Document Context:\n{context}{history_context}"
        
        response = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"{full_context}\n\nCurrent Question: {prompt}"
            }],
            model=Config.GROQ_MODEL,
            temperature=0.33,
            max_tokens=Config.MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def initialize_session():
    """Initialize session state variables"""
    defaults = {
        "processed_data": None,
        "messages": [],
        "file_uploader_key": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session():
    """Reset the application session"""
    st.session_state.processed_data = None
    st.session_state.messages = []
    st.session_state.file_uploader_key += 1
    st.rerun()

def get_conversation_history():
    """Get recent conversation history for context"""
    if not st.session_state.messages:
        return []
    
    # Get the last few exchanges (user and assistant pairs)
    history = []
    count = 0
    # Start from the end and work backwards
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            # Add the user message and the assistant response that followed it
            history.insert(0, msg)  # Add at beginning to maintain order
            count += 1
            # Stop when we reach the memory limit
            if count >= Config.MEMORY_LIMIT:
                break
    
    return history

def main():
    st.set_page_config(page_title="OCR Chatbot", layout="wide")
    st.title("📘OCR Based PDF Chatbot")
    st.caption("Analyze both text-based and scanned PDF documents with conversation memory")
    
    initialize_session()
    processor = DocumentProcessor()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Document Setup")
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=["pdf"],
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        
        st.info(f"Conversation memory: {Config.MEMORY_LIMIT} exchanges")
        
        if st.button("Clear Session"):
            reset_session()
    
    # Document processing
    if uploaded_file and not st.session_state.processed_data:
        if not Config.GROQ_API_KEY:
            st.error("Missing GROQ_API_KEY in environment variables")
            st.stop()
            
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
        
        with open(temp_pdf.name, "rb") as f:
            st.session_state.processed_data = processor.process_pdf(f.read())
        os.remove(temp_pdf.name)
        
        if st.session_state.processed_data:
            st.success(f"Processed {len(st.session_state.processed_data['texts'])} text chunks")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("references"):
                with st.expander("Document References"):
                    for ref in message["references"]:
                        st.caption(f"Page {ref['page']} ({ref['type']})")
                        st.text(ref["content"][:200] + "...")
                        st.divider()
    
    # Handle user queries
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.processed_data:
            st.error("Please upload a PDF document first")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        data = st.session_state.processed_data
        response = ""
        references = []
        
        with st.spinner("Processing your question..."):
            # Get conversation history for context
            conversation_history = get_conversation_history()
            
            # Check for page number requests
            if page_match := re.search(r"page\s+(\d+)", prompt, re.IGNORECASE):
                requested_page = int(page_match.group(1))
                if requested_page > data["max_page"]:
                    response = f"The document has {data['max_page']} pages. Page {requested_page} doesn't exist."
            
            # Process regular queries
            if not response:
                question_embed = processor.models["text_embedder"].encode([prompt])
                text_scores = cosine_similarity(question_embed, data["text_embeddings"])[0]
                top_indices = np.argsort(text_scores)[-Config.TEXT_TOP_K:][::-1]
                context = "\n".join(
                    f"Page {data['texts'][i]['page']}: {data['texts'][i]['content']}" 
                    for i in top_indices
                )
                references = [data["texts"][i] for i in top_indices]
                response = generate_response(
                    prompt, 
                    context, 
                    processor.models["groq_client"],
                    conversation_history
                )
        
        # Display response
        with st.chat_message("assistant"):
            if response:
                # Clean up response formatting
                response = re.sub(r"I (apologize|don't have|can't provide)", "The document doesn't contain", response)
                response = re.sub(r"as (an|a) AI (language )?model", "based on the document", response)
                st.markdown(response)
                
                # Store message with references
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "references": references
                })

if __name__ == "__main__":
    main()

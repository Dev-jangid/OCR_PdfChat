### on The streamlit deployment 
#### resulved version or pyterrasrect



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
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    GROQ_MODEL = "llama3-70b-8192"
    TEXT_TOP_K = 5
    MAX_TOKENS = 500
    MIN_TEXT_PER_PAGE = 50
    OCR_DPI = 200
    MEMORY_LIMIT = 5

@st.cache_resource
def load_models():
    """Load ML models and clients with error handling"""
    try:
        models = {
            "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL),
            "text_splitter": RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False
            )
        }
        
        if Config.GROQ_API_KEY:
            models["groq_client"] = Groq(api_key=Config.GROQ_API_KEY)
        else:
            st.warning("GROQ_API_KEY not found in environment variables")
            
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.error("Failed to initialize required components")
        st.stop()

class DocumentProcessor:
    def __init__(self):
        self.models = load_models()
        self._configure_ocr()
        
    def _configure_ocr(self):
        """Configure OCR based on platform"""
        if platform.system() == "Windows":
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            else:
                st.warning("Tesseract not found at default path. OCR may not work.")
        else:
            try:
                subprocess.run(['which', 'tesseract'], check=True, capture_output=True)
            except:
                st.warning("Tesseract not found in PATH. Trying to install...")
                try:
                    subprocess.run(['apt-get', 'update'], check=True)
                    subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr'], check=True)
                except Exception as e:
                    st.error(f"Failed to install Tesseract: {str(e)}")

    def process_pdf(self, file_bytes):
        """Process PDF with automatic text/OCR detection"""
        try:
            with st.spinner("Analyzing document..."):
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                texts = self._extract_text(doc)
                
                if self._needs_ocr(texts, len(doc)):
                    st.info("Scanned document detected. Running OCR...")
                    texts = self._run_ocr(doc)
                
                if not texts:
                    st.error("No readable text found")
                    return None
                    
                text_embeddings = self.models["text_embedder"].encode(
                    [t["content"] for t in texts], 
                    show_progress_bar=False
                )
                    
            return {
                "texts": texts,
                "text_embeddings": np.array(text_embeddings),
                "max_page": max(t["page"] for t in texts) if texts else 0
            }
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            st.error("Failed to process document")
            return None
    
    def _needs_ocr(self, texts, page_count):
        """Determine if OCR is needed"""
        if not texts or page_count == 0:
            return True
        total_chars = sum(len(t["content"]) for t in texts)
        return (total_chars / page_count) < Config.MIN_TEXT_PER_PAGE
    
    def _run_ocr(self, doc):
        """Robust OCR processing with fallbacks"""
        texts = []
        for page_num in range(len(doc)):
            try:
                pix = doc[page_num].get_pixmap(dpi=Config.OCR_DPI)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                img = img.convert('L')  # Convert to grayscale
                
                # Try multiple OCR approaches
                text = self._try_ocr_methods(img)
                
                if text.strip():
                    chunks = self.models["text_splitter"].split_text(text)
                    for chunk in chunks:
                        texts.append({
                            "content": chunk,
                            "page": page_num + 1,
                            "type": "ocr_text"
                        })
            except Exception as e:
                logger.warning(f"Page {page_num+1} OCR failed: {str(e)}")
                continue
        
        return texts
    
    def _try_ocr_methods(self, img):
        """Attempt different OCR approaches"""
        # Method 1: Try direct pytesseract
        try:
            return pytesseract.image_to_string(img)
        except:
            pass
        
        # Method 2: Try system tesseract command
        try:
            img_path = "/tmp/ocr_temp.png"
            img.save(img_path)
            result = subprocess.run(
                ['tesseract', img_path, 'stdout'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout
        except:
            pass
        
        return ""  # Return empty if all methods fail
    
    def _extract_text(self, doc):
        """Extract text from searchable PDF"""
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
    """Generate LLM response with enhanced error handling"""
    try:
        system_prompt = """You are a helpful document assistant. Follow these rules:
1. Answer strictly based on the provided context
2. Structure responses clearly with bullet points when appropriate
3. Never invent information not in the document
4. If unsure, say "The document doesn't contain this information"
5. Keep technical answers precise and other answers concise"""
        
        history_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversation_history[-Config.MEMORY_LIMIT:]
        ) if conversation_history else ""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nHistory:\n{history_context}\n\nQuestion: {prompt}"}
            ],
            model=Config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=Config.MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return "Sorry, I encountered an error processing your request."

def initialize_session():
    """Initialize session state with defaults"""
    defaults = {
        "processed_data": None,
        "messages": [],
        "file_uploader_key": 0
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def reset_session():
    """Reset conversation while keeping models loaded"""
    st.session_state.processed_data = None
    st.session_state.messages = []
    st.session_state.file_uploader_key += 1
    st.rerun()

def main():
    st.set_page_config(page_title="Document AI Assistant", layout="wide")
    st.title("📄 Smart Document Analyzer")
    st.caption("Extract insights from both digital and scanned PDFs")
    
    initialize_session()
    processor = DocumentProcessor()
    
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=["pdf"],
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        st.download_button(
            "Download Sample PDF",
            data=open("sample.pdf", "rb").read() if os.path.exists("sample.pdf") else b"",
            file_name="sample.pdf",
            disabled=not os.path.exists("sample.pdf")
        )
        st.button("New Conversation", on_click=reset_session)
    
    if uploaded_file and not st.session_state.processed_data:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as f:
                st.session_state.processed_data = processor.process_pdf(f.read())
        finally:
            os.unlink(tmp_path)
        
        if st.session_state.processed_data:
            st.success(f"Processed {len(st.session_state.processed_data['texts'])} text segments")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("references"):
                with st.expander("Source Excerpts"):
                    for ref in message["references"]:
                        st.caption(f"Page {ref['page']} ({ref['type']})")
                        st.text(ref["content"][:250] + ("..." if len(ref["content"]) > 250 else ""))
    
    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.processed_data:
            st.error("Please upload a document first")
            return
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Generating response..."):
            data = st.session_state.processed_data
            conversation_history = [
                m for m in st.session_state.messages 
                if m["role"] in ("user", "assistant")
            ][-Config.MEMORY_LIMIT*2:]
            
            if re.search(r"page\s+(\d+)", prompt, re.IGNORECASE):
                page_match = re.search(r"page\s+(\d+)", prompt, re.IGNORECASE)
                requested_page = int(page_match.group(1))
                if requested_page > data["max_page"]:
                    response = f"The document only has {data['max_page']} pages."
                else:
                    page_texts = [t for t in data["texts"] if t["page"] == requested_page]
                    response = "\n\n".join(t["content"] for t in page_texts)
                    references = page_texts
            else:
                question_embed = processor.models["text_embedder"].encode([prompt])
                similarities = cosine_similarity(question_embed, data["text_embeddings"])[0]
                top_indices = np.argsort(similarities)[-Config.TEXT_TOP_K:][::-1]
                context = "\n".join(
                    f"Page {data['texts'][i]['page']}:\n{data['texts'][i]['content']}" 
                    for i in top_indices
                )
                references = [data["texts"][i] for i in top_indices]
                response = generate_response(
                    prompt,
                    context,
                    processor.models["groq_client"],
                    conversation_history
                )
        
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "references": references
            })

if __name__ == "__main__":
    main()

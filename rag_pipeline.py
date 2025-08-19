# rag_pipeline.py
import os, re, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# ---- loaders ----
import fitz  # PyMuPDF for PDFs (imported as 'fitz')
import docx  # python-docx
from pptx import Presentation  # python-pptx

# ---- vector store ----
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---- http for Ollama / optional OpenAI ----
import requests


# ---------------------------
# Text utils
# ---------------------------
def normalize_ws(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ---------------------------
# File loaders
# ---------------------------
def load_pdf(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    out, doc = [], fitz.open(path)
    for i, page in enumerate(doc):
        text = normalize_ws(page.get_text("text") or "")
        if text:
            out.append((text, {"type": "pdf", "page": i + 1}))
    doc.close()
    return out

def load_docx(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    d = docx.Document(path)
    text = normalize_ws("\n".join(p.text for p in d.paragraphs))
    return [(text, {"type": "docx"})] if text else []

def load_pptx(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    out, pres = [], Presentation(path)
    for i, slide in enumerate(pres.slides):
        parts = []
        for shp in slide.shapes:
            if hasattr(shp, "text") and shp.text:
                parts.append(shp.text)
        text = normalize_ws("\n".join(parts))
        if text:
            out.append((text, {"type": "pptx", "slide": i + 1}))
    return out

def load_file(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":  return load_pdf(path)
    if ext == ".docx": return load_docx(path)
    if ext == ".pptx": return load_pptx(path)
    return []


# ---------------------------
# Embeddings
# ---------------------------
class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text:latest", base_url: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        vecs: List[List[float]] = []
        for t in texts:
            try:
                resp = requests.post(url, json={"model": self.model, "prompt": t}, timeout=self.timeout)
                resp.raise_for_status()
            except requests.HTTPError as e:
                if getattr(e, "response", None) is not None and e.response.status_code == 404:
                    raise RuntimeError(
                        "Ollama embeddings endpoint (/api/embeddings) not found. "
                        "Upgrade Ollama and pull an embedding model, e.g.: "
                        "`ollama pull nomic-embed-text:latest`"
                    ) from e
                raise
            data = resp.json()
            v = data.get("embedding")
            if not isinstance(v, list):
                raise RuntimeError("Invalid embedding response from Ollama.")
            vecs.append(v)
        return vecs


class OpenAIEmbeddings:
    def __init__(self, model: str = "text-embedding-3-small", dimensions: Optional[int] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(model=self.model, input=texts, **kwargs)
        return [d.embedding for d in resp.data]


# ---------------------------
# Vector DB (Chroma)
# ---------------------------
@dataclass
class ChromaConfig:
    path: str = "chroma_db"
    collection: str = "nps_cyber_ta"

class ChromaStore:
    def __init__(self, cfg: ChromaConfig, ef=None):
        os.makedirs(cfg.path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=cfg.path, settings=Settings(allow_reset=False))
        self.collection = self.client.get_or_create_collection(name=cfg.collection, embedding_function=ef)

    def add_with_embeddings(self, ids, docs, metadatas, embeddings):
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    def add_docs(self, ids, docs, metadatas):
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas)

    def query(self, query_texts=None, query_embeddings=None, n_results=4, where=None):
        kwargs = {"n_results": n_results}
        if where: kwargs["where"] = where
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        elif query_texts is not None:
            kwargs["query_texts"] = query_texts
        else:
            raise ValueError("Provide query_texts or query_embeddings")
        return self.collection.query(**kwargs)


# ---------------------------
# RAG Pipeline
# ---------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

@dataclass
class RAGConfig:
    repo_dir: str
    chroma_path: str = "chroma_db"
    collection: str = "nps_cyber_ta"          # base name (we append a suffix automatically)
    embed_provider: str = "ollama"            # "ollama" | "openai" | "onnx"
    embed_model: str = "nomic-embed-text:latest"
    embed_dimensions: Optional[int] = None    # only for openai
    ollama_url: str = "http://localhost:11434"
    chunk_size: int = 1000
    chunk_overlap: int = 200

class RAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

        ef = None
        # decide embedding + canonical dimension (for suffix)
        if cfg.embed_provider == "openai":
            self.embedder = OpenAIEmbeddings(model=cfg.embed_model, dimensions=cfg.embed_dimensions)
            dim = cfg.embed_dimensions or (1536 if "text-embedding-3-small" in cfg.embed_model else 3072)
        elif cfg.embed_provider == "onnx":
            ef = embedding_functions.ONNXMiniLM_L6_V2()  # 384-d
            self.embedder = None
            dim = 384
        else:
            self.embedder = OllamaEmbeddings(model=cfg.embed_model, base_url=cfg.ollama_url)
            dim = None  # unknown/varies; we isolate by provider+model

        suffix = f"{_slug(cfg.embed_provider)}-{_slug(cfg.embed_model)}-{dim or 'na'}"
        coll_name = f"{cfg.collection}-{suffix}"

        self.store = ChromaStore(ChromaConfig(path=cfg.chroma_path, collection=coll_name), ef=ef)

    def _gen_id(self, path: str, chunk_index: int, salt: str) -> str:
        h = hashlib.sha1()
        h.update(path.encode("utf-8"))
        h.update(str(chunk_index).encode("utf-8"))
        h.update(salt.encode("utf-8"))
        return h.hexdigest()

    def index_repo(self) -> Dict[str, Any]:
        repo = os.path.abspath(self.cfg.repo_dir)
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"Repo directory not found: {repo}")

        supported = (".pdf", ".docx", ".pptx")
        files = []
        for root, _, filenames in os.walk(repo):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in supported:
                    files.append(os.path.join(root, fn))
        files.sort()

        added = 0
        for path in files:
            try:
                stat = os.stat(path)
                salt = str(int(stat.st_mtime))
                records = load_file(path)
                for text, meta in records:
                    chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
                    if not chunks:
                        continue
                    ids = [self._gen_id(path, i, salt) for i in range(len(chunks))]
                    metas = []
                    for i in range(len(chunks)):
                        md = dict(meta)
                        md.update({
                            "source": os.path.relpath(path, repo).replace("\\", "/"),
                            "chunk_index": i,
                        })
                        metas.append(md)
                    if self.store.collection._embedding_function is not None:
                        # ONNX via Chroma (no external calls)
                        self.store.add_docs(ids=ids, docs=chunks, metadatas=metas)
                    else:
                        # External embedder (Ollama or OpenAI)
                        embeds = self.embedder.embed(chunks)
                        self.store.add_with_embeddings(ids=ids, docs=chunks, metadatas=metas, embeddings=embeds)
                    added += len(chunks)
            except Exception as e:
                print(f"[index] Error {path}: {e}")
                continue

        return {"indexed_files": len(files), "indexed_chunks": added}

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        # If the collection has no embedding_function, embed the query ourselves
        has_ef = getattr(self.store.collection, "_embedding_function", None) is not None
        if not has_ef and getattr(self, "embedder", None) is not None:
            q_vec = self.embedder.embed([query])
            res = self.store.query(query_embeddings=q_vec, n_results=k)
        else:
            res = self.store.query(query_texts=[query], n_results=k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        out = []
        for doc, md, id_ in zip(docs, metas, ids):
            src = md.get("source", "unknown")
            loc = md.get("page") or md.get("slide") or md.get("chunk_index")
            out.append({"id": id_, "text": doc, "source": src, "loc": loc})
        return out

    @staticmethod
    def format_context(snippets: List[Dict[str, Any]]) -> str:
        parts = []
        for i, s in enumerate(snippets, 1):
            loc_str = f"{s['source']}#{s['loc']}"
            txt = s["text"]
            if len(txt) > 800:
                txt = txt[:800] + " ..."
            parts.append(f"[{i}] ({loc_str})\n{txt}")
        return "Use the following retrieved context to answer. If relevant, cite with [#] or (source#loc).\n\n" + "\n\n".join(parts)

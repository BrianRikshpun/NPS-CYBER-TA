# app.py
import os, io, base64, json, re
from typing import List, Dict, Any, Union, Tuple
from PIL import Image
import gradio as gr

from agent import Agent, AgentConfig, ChatMessage
from backends import make_backend
from rag_pipeline import RAGPipeline, RAGConfig


# ---------------- Header / Logo ----------------
CUSTOM_CSS = """
#header { position: relative; display: flex; align-items: center; justify-content: center;
  padding: 12px 16px; border-bottom: 1px solid var(--border-color-primary);
  background: var(--body-background-fill); }
#header .logo { position: absolute; left: 16px; height: 48px; width: auto; }
#header .title { font-size: 28px; font-weight: 800; letter-spacing: 1px; }
@media (max-width: 640px) { #header .title { font-size: 20px; } #header .logo { height: 36px; } }
"""
def _logo_data_uri() -> str:
    local_path = os.getenv("NPS_LOGO_PATH", "assets/nps_logo.png")
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    return os.getenv("NPS_LOGO_URL", "")


# ---------------- Image agent ----------------
class ImageAgent:
    def __init__(self, model: str = "gpt-image-1", size: str = "1024x1024"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment for image generation.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
    def generate(self, prompt: str) -> Image.Image:
        if not prompt or not prompt.strip():
            raise ValueError("Empty image prompt.")
        resp = self.client.images.generate(model=self.model, prompt=prompt, size=self.size, n=1)
        raw = base64.b64decode(resp.data[0].b64_json)
        return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------- Helpers ----------------
def is_image_request(text: str) -> Tuple[bool, str]:
    if not text:
        return False, text
    s = text.strip(); l = s.lower()
    for p in ("/image", "image:", "img:", "/img"):
        if l.startswith(p):
            return True, s[len(p):].strip(" :")
    return False, text

def build_chat_agent(provider: str, ollama_model: str, openai_model: str, temperature: float, max_tokens: Union[int, None], ollama_url: str) -> Agent:
    cfg = AgentConfig(
        provider=("ollama" if provider == "ollama" else "openai"),
        model=(ollama_model if provider == "ollama" else openai_model),
        temperature=temperature, max_tokens=max_tokens,
        system_prompt="You are a concise, helpful teaching assistant. Prefer stepwise reasoning and cite equations when relevant.",
        ollama_base_url=ollama_url,
    )
    return Agent(make_backend(cfg), cfg)

def user_msg(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}
def assistant_text(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


# --- Auto-dimension + collection preview for RAG ---
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

BASE_COLLECTION = "nps_cyber_ta"

def default_openai_dims(model: str) -> int:
    m = (model or "").lower()
    if "text-embedding-3-large" in m:
        return 3072
    return 1536  # small

def derive_auto_dims(provider: str, model: str) -> Union[int, None]:
    if provider == "openai":
        return default_openai_dims(model)
    if provider == "onnx":
        return 384
    return None  # ollama

def derive_collection_name(provider: str, model: str, dims: Union[int, None]) -> str:
    return f"{BASE_COLLECTION}-{_slug(provider)}-{_slug(model or 'model')}-{dims or 'na'}"


# ---------------- Core handler (with RAG) ----------------
def chat_or_image(
    user_text: str,
    display_history: List[Dict[str, Any]],
    llm_context: List[ChatMessage],
    provider: str,
    ollama_model: str,
    openai_model: str,
    temperature: float,
    max_tokens: Union[int, None],
    img_size: str,
    # NEW: Ollama URL from UI
    ollama_url: str,
    # RAG params:
    rag_enabled: bool,
    repo_dir: str,
    chroma_path: str,
    embed_provider: str,
    embed_model: str,
    embed_dimensions_auto: Union[int, None],
    top_k: int,
    rag_state: Union[None, str],
):
    user_text = user_text or ""
    wants_image, prompt = is_image_request(user_text)

    display_history = display_history or []
    llm_context = llm_context or []

    display_history.append(user_msg(user_text))
    llm_context.append(ChatMessage(role="user", content=user_text))

    if wants_image:
        try:
            img_agent = ImageAgent(model="gpt-image-1", size=img_size)
            img = img_agent.generate(prompt)
            note = f"[Generated image for: {prompt}]"
            display_history.append(assistant_text(note))
            llm_context.append(ChatMessage(role="assistant", content=note))
            return display_history, img, display_history, llm_context, rag_state
        except Exception as e:
            err = f"Image error: {e}"
            display_history.append(assistant_text(err))
            llm_context.append(ChatMessage(role="assistant", content=err))
            return display_history, None, display_history, llm_context, rag_state

    # RAG retrieval
    rag_system_context: List[ChatMessage] = []
    if rag_enabled:
        try:
            dims = embed_dimensions_auto if embed_provider == "openai" else (384 if embed_provider == "onnx" else None)
            cfg_dict = {
                "repo_dir": repo_dir,
                "chroma_path": chroma_path,
                "collection": BASE_COLLECTION,
                "embed_provider": embed_provider,
                "embed_model": embed_model,
                "embed_dimensions": dims,
                "ollama_url": ollama_url,  # use the same URL for embeddings
            }
            pipeline = RAGPipeline(RAGConfig(**cfg_dict))
            snippets = pipeline.retrieve(user_text, k=max(1, int(top_k or 4)))
            ctx_block = RAGPipeline.format_context(snippets) if snippets else "No relevant context found."
            rag_system_context = [ChatMessage(role="system", content=ctx_block)]
            rag_state = json.dumps(cfg_dict)
        except Exception as e:
            display_history.append(assistant_text(f"[RAG error: {e} — continuing without RAG]"))

    # Chat
    agent = build_chat_agent(provider, ollama_model, openai_model, temperature, max_tokens, ollama_url)
    try:
        context_msgs = rag_system_context + llm_context[:-1]
        reply = agent.chat(user_text, context=context_msgs, stream=False)
        display_history.append(assistant_text(reply))
        llm_context.append(ChatMessage(role="assistant", content=reply))
        return display_history, None, display_history, llm_context, rag_state
    except Exception as e:
        err = f"Chat error: {e}"
        display_history.append(assistant_text(err))
        llm_context.append(ChatMessage(role="assistant", content=err))
        return display_history, None, display_history, llm_context, rag_state


def rebuild_index(repo_dir: str, chroma_path: str, embed_provider: str, embed_model: str, embed_dimensions_auto: Union[int, None], ollama_url: str):
    try:
        dims = embed_dimensions_auto if embed_provider == "openai" else (384 if embed_provider == "onnx" else None)
        cfg = RAGConfig(
            repo_dir=repo_dir, chroma_path=chroma_path,
            collection=BASE_COLLECTION,
            embed_provider=embed_provider, embed_model=embed_model,
            embed_dimensions=dims, ollama_url=ollama_url,
        )
        pipeline = RAGPipeline(cfg)
        stats = pipeline.index_repo()
        cfg_json = json.dumps({
            "repo_dir": repo_dir, "chroma_path": chroma_path,
            "collection": BASE_COLLECTION,
            "embed_provider": embed_provider, "embed_model": embed_model,
            "embed_dimensions": dims, "ollama_url": ollama_url,
        })
        log = f"✅ Indexed {stats['indexed_files']} files, {stats['indexed_chunks']} chunks."
        return log, cfg_json
    except Exception as e:
        return f"❌ Index error: {e}", None


# ---------------- UI ----------------
logo_src = _logo_data_uri()

with gr.Blocks(title="NPS - CYBER - TA", css=CUSTOM_CSS) as demo:
    gr.HTML(
        f"""
        <div id="header">
          {f'<img class="logo" src="{logo_src}" alt="Naval Postgraduate School Logo" />' if logo_src else ''}
          <div class="title">NPS - CYBER - TA</div>
        </div>
        """
    )

    # Chat params (added Ollama Base URL)
    with gr.Row():
        provider = gr.Radio(choices=["ollama", "openai"], value="ollama", label="Chat provider", scale=1)
        ollama_model = gr.Textbox(value="llama3.2:1b-instruct-q4_K_M", label="Ollama model", scale=2)
        ollama_url = gr.Textbox(value="http://127.0.0.1:11434", label="Ollama Base URL", scale=2)  # NEW
        openai_model = gr.Dropdown(choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], value="gpt-4o-mini", label="OpenAI model", scale=2)
        temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature", scale=1)
        max_tokens = gr.Number(value=None, precision=0, label="Max new tokens", scale=1)
        img_size = gr.Dropdown(choices=["512x512", "768x768", "1024x1024"], value="1024x1024", label="Image size", scale=1)

    # RAG controls
    with gr.Accordion("RAG (Local Retrieval Augmented Generation)", open=False):
        with gr.Row():
            rag_enabled = gr.Checkbox(value=False, label="Enable RAG")
            repo_dir = gr.Textbox(value="docs", label="Docs folder (PDF/DOCX/PPTX)")
            chroma_path = gr.Textbox(value="chroma_db", label="Chroma DB path")
        with gr.Row():
            embed_provider = gr.Radio(choices=["ollama", "openai", "onnx"], value="ollama", label="Embedding provider")
            embed_model = gr.Textbox(value="nomic-embed-text:latest", label="Embedding model")
            embed_dimensions_auto = gr.Textbox(value="", label="Embedding dim (auto)", interactive=False)
            collection_preview = gr.Textbox(value="", label="Collection (auto)", interactive=False)
            top_k = gr.Slider(1, 10, value=4, step=1, label="Top-K")
        with gr.Row():
            reindex_btn = gr.Button("Rebuild Index", variant="secondary")
            reindex_log = gr.Markdown("")
    rag_state = gr.State(None)

    # Main content
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, type="messages", label="Chat Assistant")
        with gr.Column(scale=2):
            image_output = gr.Image(height=520, label="Image Generator")

    with gr.Row():
        user_box = gr.Textbox(placeholder="Type a message. For images, use `/image <prompt>`", label="Your message", scale=5)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    # States
    history_state = gr.State([])   # Chatbot messages
    messages_state = gr.State([])  # ChatMessage list

    # --- live auto updates for dim + collection preview ---
    def _auto_update(provider_val, model_val):
        dims = derive_auto_dims(provider_val, model_val)
        dims_str = "" if dims is None else str(dims)
        coll = derive_collection_name(provider_val, model_val, dims)
        return dims_str, coll

    embed_provider.change(_auto_update, inputs=[embed_provider, embed_model], outputs=[embed_dimensions_auto, collection_preview])
    embed_model.change(_auto_update, inputs=[embed_provider, embed_model], outputs=[embed_dimensions_auto, collection_preview])
    demo.load(_auto_update, inputs=[embed_provider, embed_model], outputs=[embed_dimensions_auto, collection_preview])

    # Actions
    def _submit(user_text, hist_msgs, llm_msgs, provider, ollama_model, openai_model, temperature, max_tokens, img_size,
                ollama_url, rag_enabled, repo_dir, chroma_path, embed_provider, embed_model, embed_dimensions_auto, top_k, rag_state):
        dims = None if not embed_dimensions_auto else int(embed_dimensions_auto)
        return chat_or_image(
            user_text, hist_msgs, llm_msgs,
            provider, ollama_model, openai_model, temperature, max_tokens, img_size,
            ollama_url,
            rag_enabled, repo_dir, chroma_path, embed_provider, embed_model, dims, top_k, rag_state
        )

    send_btn.click(
        _submit,
        inputs=[user_box, history_state, messages_state, provider, ollama_model, openai_model, temperature, max_tokens, img_size,
                ollama_url, rag_enabled, repo_dir, chroma_path, embed_provider, embed_model, embed_dimensions_auto, top_k, rag_state],
        outputs=[chatbot, image_output, history_state, messages_state, rag_state],
    ).then(lambda: "", outputs=user_box)

    user_box.submit(
        _submit,
        inputs=[user_box, history_state, messages_state, provider, ollama_model, openai_model, temperature, max_tokens, img_size,
                ollama_url, rag_enabled, repo_dir, chroma_path, embed_provider, embed_model, embed_dimensions_auto, top_k, rag_state],
        outputs=[chatbot, image_output, history_state, messages_state, rag_state],
    ).then(lambda: "", outputs=user_box)

    clear_btn.click(lambda: ([], None, [], [], None), outputs=[chatbot, image_output, history_state, messages_state, rag_state])

    def _reindex(repo_dir, chroma_path, embed_provider, embed_model, embed_dimensions_auto, ollama_url):
        dims = None if not embed_dimensions_auto else int(embed_dimensions_auto)
        return rebuild_index(repo_dir, chroma_path, embed_provider, embed_model, dims, ollama_url)

    reindex_btn.click(
        _reindex,
        inputs=[repo_dir, chroma_path, embed_provider, embed_model, embed_dimensions_auto, ollama_url],
        outputs=[reindex_log, rag_state]
    )

if __name__ == "__main__":
    demo.launch()

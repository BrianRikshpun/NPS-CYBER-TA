# app.py
import os
import io
import base64
from typing import List, Tuple, Any, Dict, Union
from PIL import Image
import gradio as gr

from agent import Agent, AgentConfig, ChatMessage
from backends import make_backend

# -------------------------------------------------
# Header config (logo + centered title)
# -------------------------------------------------
LOGO_URL = os.getenv(
    "NPS_LOGO_URL",
    "https://upload.wikimedia.org/wikipedia/en/1/1b/Naval_Postgraduate_School_seal.png"
)

CUSTOM_CSS = """
#header {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center; /* centers the title */
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color-primary);
  background: var(--body-background-fill);
}
#header .logo {
  position: absolute; /* pin to left while title stays centered */
  left: 16px;
  height: 48px;
  width: auto;
}
#header .title {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 1px;
}
@media (max-width: 640px) {
  #header .title { font-size: 20px; }
  #header .logo { height: 36px; }
}
"""

# -----------------------------
# Image Agent (OpenAI DALL·E / gpt-image-1)
# -----------------------------
class ImageAgent:
    """
    Uses OpenAI Images API (DALL·E via gpt-image-1) to generate a single image.
    Requires OPENAI_API_KEY in environment and (often) org verification.
    """
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
        resp = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            n=1,
        )
        b64 = resp.data[0].b64_json
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")


# -----------------------------
# Helpers
# -----------------------------
def is_image_request(text: str) -> Tuple[bool, str]:
    """
    Only generate images when explicitly requested.
    Prefixes:
      /image <prompt>  |  image: <prompt>  |  img: <prompt>  |  /img <prompt>
    """
    if not text:
        return False, text
    stripped = text.strip()
    lowered = stripped.lower()
    prefixes = ("/image", "image:", "img:", "/img")
    for p in prefixes:
        if lowered.startswith(p):
            prompt = stripped[len(p):].strip(" :")
            return True, (prompt if prompt else stripped)
    return False, text


def build_chat_agent(
    provider: str,
    ollama_model: str,
    openai_model: str,
    temperature: float,
    max_tokens: Union[int, None],
) -> Agent:
    cfg = AgentConfig(
        provider=("ollama" if provider == "ollama" else "openai"),
        model=(ollama_model if provider == "ollama" else openai_model),
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt="You are a concise, helpful teaching assistant. Prefer stepwise reasoning and cite equations when relevant.",
    )
    return Agent(make_backend(cfg), cfg)


def build_image_agent(size: str = "1024x1024") -> ImageAgent:
    return ImageAgent(model="gpt-image-1", size=size)


# ------ Chatbot (messages) helpers ------
def user_msg(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}

def assistant_text(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


# -----------------------------
# Main handler
# Returns FOUR outputs to match: [chatbot, image, history_state, messages_state]
# -----------------------------
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
):
    user_text = user_text or ""
    wants_image, prompt = is_image_request(user_text)

    display_history = display_history or []
    llm_context = llm_context or []

    # Append user's turn
    display_history.append(user_msg(user_text))
    llm_context.append(ChatMessage(role="user", content=user_text))

    if wants_image:
        try:
            img_agent = build_image_agent(size=img_size)
            img = img_agent.generate(prompt)
            note = f"[Generated image for: {prompt}]"
            display_history.append(assistant_text(note))
            llm_context.append(ChatMessage(role="assistant", content=note))
            return display_history, img, display_history, llm_context
        except Exception as e:
            err = f"Image error: {e}"
            display_history.append(assistant_text(err))
            llm_context.append(ChatMessage(role="assistant", content=err))
            return display_history, None, display_history, llm_context

    # Otherwise chat
    agent = build_chat_agent(
        provider=provider,
        ollama_model=ollama_model,
        openai_model=openai_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        reply = agent.chat(user_text, context=llm_context[:-1], stream=False)
        display_history.append(assistant_text(reply))
        llm_context.append(ChatMessage(role="assistant", content=reply))
        return display_history, None, display_history, llm_context
    except Exception as e:
        err = f"Chat error: {e}"
        display_history.append(assistant_text(err))
        llm_context.append(ChatMessage(role="assistant", content=err))
        return display_history, None, display_history, llm_context


def clear_all():
    # chatbot messages, image, chatbot state (same), llm state
    return [], None, [], []


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Multi-Agent Chat (Ollama + DALL·E)", css=CUSTOM_CSS) as demo:
    # ====== HEADER ======
    gr.HTML(
        f"""
        <div id="header">
          <img class="logo" src="{LOGO_URL}" alt="Naval Postgraduate School Logo" />
          <div class="title">NPS - CYBER - TA</div>
        </div>
        """
    )

    # ====== PARAMETERS (TOP) ======
    with gr.Row():
        provider = gr.Radio(choices=["ollama", "openai"], value="ollama", label="Chat provider", scale=1)
        ollama_model = gr.Textbox(value="llama3.2:3b-instruct-q4_K_M", label="Ollama model", scale=2)
        openai_model = gr.Dropdown(
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            value="gpt-4o-mini",
            label="OpenAI model",
            scale=2
        )
        temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature", scale=1)
        max_tokens = gr.Number(value=None, precision=0, label="Max new tokens", scale=1)
        img_size = gr.Dropdown(choices=["512x512", "768x768", "1024x1024"], value="1024x1024", label="Image size", scale=1)

    # ====== MAIN CONTENT (SIDE BY SIDE) ======
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, type="messages", label="Chat Assistant")
        with gr.Column(scale=2):
            image_output = gr.Image(height=520, label="Image Generator")

    # ====== INPUT / CONTROLS ======
    with gr.Row():
        user_box = gr.Textbox(placeholder="Type a message. For images, use `/image <prompt>`", label="Your message", scale=5)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    # States mirror visible components
    history_state = gr.State([])   # messages-format list for Chatbot
    messages_state = gr.State([])  # List[ChatMessage] for LLM context

    def _submit(user_text, hist_msgs, llm_msgs, provider, ollama_model, openai_model, temperature, max_tokens, img_size):
        return chat_or_image(
            user_text, hist_msgs, llm_msgs,
            provider, ollama_model, openai_model,
            temperature, max_tokens, img_size
        )

    # Wire up actions
    send_btn.click(
        _submit,
        inputs=[user_box, history_state, messages_state, provider, ollama_model, openai_model, temperature, max_tokens, img_size],
        outputs=[chatbot, image_output, history_state, messages_state],
    ).then(lambda: "", outputs=user_box)

    user_box.submit(
        _submit,
        inputs=[user_box, history_state, messages_state, provider, ollama_model, openai_model, temperature, max_tokens, img_size],
        outputs=[chatbot, image_output, history_state, messages_state],
    ).then(lambda: "", outputs=user_box)

    clear_btn.click(
        clear_all,
        outputs=[chatbot, image_output, history_state, messages_state]
    )

if __name__ == "__main__":
    demo.launch(share=True)

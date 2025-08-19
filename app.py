# app.py
import io
import os
import base64
from typing import List, Tuple, Any, Dict, Union
from PIL import Image
import gradio as gr

from agent import Agent, AgentConfig, ChatMessage
from backends import make_backend

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
        b64 = resp.data[0].b64_json  # base64-encoded PNG
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")


# -----------------------------------------
# Helpers
# -----------------------------------------
def is_image_request(text: str) -> Tuple[bool, str]:
    """
    Only generate images when explicitly requested.
    Supported prefixes:
      /image <prompt>   |  image: <prompt>  |  img: <prompt>  |  /img <prompt>
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
    temperature: float,
    max_tokens: Union[int, None],
) -> Agent:
    cfg = AgentConfig(
        provider=("ollama" if provider == "ollama" else "openai"),
        model=(ollama_model if provider == "ollama" else "gpt-4o-mini"),
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt="You are a concise, helpful teaching assistant. Prefer stepwise reasoning and cite equations when relevant.",
    )
    return Agent(make_backend(cfg), cfg)


def build_image_agent(size: str = "1024x1024") -> ImageAgent:
    return ImageAgent(model="gpt-image-1", size=size)


# -------------- Chatbot (messages) helpers --------------
def user_msg(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}

def assistant_text(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


# -----------------------------------------
# Main handler
#   - display_history: List[dict] with {'role':..., 'content':...}  (for Chatbot type="messages")
#   - llm_context:     List[ChatMessage] used by Agent for context
# Returns FOUR values to match outputs=[chatbot, image_output, history_state, messages_state]
# -----------------------------------------
def chat_or_image(
    user_text: str,
    display_history: List[Dict[str, Any]],
    llm_context: List[ChatMessage],
    ollama_model: str,
    temperature: float,
    max_tokens: Union[int, None],
    provider: str,
    img_size: str,
):
    user_text = user_text or ""
    wants_image, prompt = is_image_request(user_text)

    # Ensure states
    display_history = display_history or []
    llm_context = llm_context or []

    # Append user to both histories
    display_history.append(user_msg(user_text))
    llm_context.append(ChatMessage(role="user", content=user_text))

    if wants_image:
        # IMAGE path: generate image; add a text breadcrumb to the chat
        try:
            img_agent = build_image_agent(size=img_size)
            img = img_agent.generate(prompt)
            note = f"[Generated image for: {prompt}]"
            display_history.append(assistant_text(note))
            llm_context.append(ChatMessage(role="assistant", content=note))
            # Return 4 values: chatbot msgs, image, chatbot state, llm state
            return display_history, img, display_history, llm_context
        except Exception as e:
            err = f"Image error: {e}"
            display_history.append(assistant_text(err))
            llm_context.append(ChatMessage(role="assistant", content=err))
            return display_history, None, display_history, llm_context

    # Otherwise: normal chat
    agent = build_chat_agent(provider=provider, ollama_model=ollama_model, temperature=temperature, max_tokens=max_tokens)
    try:
        # Provide prior context except the latest user (Agent.chat appends user internally)
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
    # chatbot messages, image, chatbot state (same as messages), llm state
    return [], None, [], []


# -----------------------------------------
# UI
# -----------------------------------------
with gr.Blocks(title="Multi-Agent Chat (Ollama + DALL·E)") as demo:
    gr.Markdown(
        """
        # 🧑‍🏫 Multi-Agent Chat (Open-Source + DALL·E)
        - **Chat agent**: open-source via **Ollama** (e.g., Llama 3).  
        - **Image agent**: **OpenAI DALL·E** (via `gpt-image-1`).  
        - To generate an image, **explicitly** ask using one of:
          - `/image a diagram of bias–variance tradeoff`
          - `image: a logo with a rocket and book`
          - `img: a cute llama wearing glasses`
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            # Use messages-format (OpenAI style)
            chatbot = gr.Chatbot(height=520, type="messages")
            image_output = gr.Image(height=520, label="Generated Image", visible=True)
            user_box = gr.Textbox(placeholder="Type a message. For images, use `/image <prompt>`", label="Your message")
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            gr.Markdown("### Settings")
            provider = gr.Radio(choices=["ollama", "openai"], value="ollama", label="Chat provider")
            ollama_model = gr.Textbox(value="llama3:latest", label="Ollama model (chat)")
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature (chat)")
            max_tokens = gr.Number(value=None, precision=0, label="Max new tokens (chat) — optional")
            img_size = gr.Dropdown(choices=["512x512", "768x768", "1024x1024"], value="1024x1024", label="Image size")

            gr.Markdown(
                """
                **Requirements**
                - For **Ollama chat**: install [Ollama](https://ollama.com), run `ollama pull llama3:latest`.
                - For **images**: set `OPENAI_API_KEY` and ensure your org is verified for `gpt-image-1`.
                """
            )

    # States:
    # - history_state (messages-format) mirrors what Chatbot shows
    # - messages_state (ChatMessage list) is used as LLM context
    history_state = gr.State([])      # List[dict]: [{"role": "...", "content": "..."}]
    messages_state = gr.State([])     # List[ChatMessage]

    def _submit(user_text, hist_msgs, llm_msgs, ollama_model, temperature, max_tokens, provider, img_size):
        # MUST return 4 values to match outputs
        return chat_or_image(user_text, hist_msgs, llm_msgs, ollama_model, temperature, max_tokens, provider, img_size)

    # Send button
    send_btn.click(
        _submit,
        inputs=[user_box, history_state, messages_state, ollama_model, temperature, max_tokens, provider, img_size],
        outputs=[chatbot, image_output, history_state, messages_state],
    ).then(lambda: "", outputs=user_box)

    # Enter key
    user_box.submit(
        _submit,
        inputs=[user_box, history_state, messages_state, ollama_model, temperature, max_tokens, provider, img_size],
        outputs=[chatbot, image_output, history_state, messages_state],
    ).then(lambda: "", outputs=user_box)

    # Clear
    clear_btn.click(
        clear_all,
        outputs=[chatbot, image_output, history_state, messages_state]
    )

if __name__ == "__main__":
    demo.launch()

import os
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, request, session, render_template, redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------- Load Environment ----------------------
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

client = OpenAI()  # Uses OPENAI_API_KEY from .env
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
VECTOR_STORE_BRD = os.environ.get("VECTOR_STORE_BRD", "").strip()
VECTOR_STORE_FSD = os.environ.get("VECTOR_STORE_FSD", "").strip()


# ---------------------- Helpers ----------------------
def _ensure_history():
    """Initialize a simple in-session chat history."""
    if "chat" not in session:
        session["chat"] = []  # list of dicts: {"role": "user"/"assistant", "text": "...", "time": "...", "response_id": "..."}
    return session["chat"]


def _ensure_source():
    """Initialize default source (BRD)."""
    if "source" not in session:
        session["source"] = "brd"  # default to BRD
    return session["source"]


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _iso_to_local(ts: str) -> str:
    """
    Convert ISO timestamp (UTC) to Qatar local time (UTC+3)
    and render it as a readable local time label.
    """
    try:
        dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        qatar_tz = timezone(timedelta(hours=3))
        dt_qatar = dt_utc.astimezone(qatar_tz)
        return dt_qatar.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _get_system_prompt(source):
    """Generate dynamic system prompt based on selected source."""
    if source == "brd":
        doc_type = "Business Requirements Documents (BRDs) in Arabic"
        scope_msg = "This question appears to be outside the scope of the BRDs."
    else:  # fsd
        doc_type = "Functional Specification Documents (FSDs) in English"
        scope_msg = "This question appears to be outside the scope of the FSDs."

    return f"""You are an AI assistant specialized in answering questions only from official {doc_type}.

Role:
Your core purpose is to help users understand system workflows, processes, business rules, and requirements extracted from those documents.
You must not answer questions outside the document domain.

All factual answers must come exclusively from the indexed documents.

Always reply in the same language as the user's question.

Provide a clear, structured answer.

Always include citations from the retrieved documents (if available) — specify:
- Document name
- Section name or heading
- Page number

Be concise but faithful — summarize multiple chunks logically without changing their meaning.

Never guess or assume. If no direct match is found, say clearly:
"{scope_msg}"

Support follow-up questions — retain context across turns within the session.

If multiple related sections are found, merge insights and list all relevant sources.

Maintain a professional, explanatory tone suitable for analysts or technical users.

If the query concerns anything outside the document domain (e.g., religious, legal, personal, or unrelated technical questions):
"I can only answer questions that are documented in the official documents."

If unsure, use:
"{scope_msg}"

Preserve the chat context to understand follow-up questions.

If the user asks a follow-up like "What happens next?" — infer based on previous question context and retrieved document continuity.
"""


def _extract_text_and_sources(resp):
    text = getattr(resp, "output_text", "") or ""
    sources = []
    try:
        for item in resp.output or []:
            for block in getattr(item, "content", []) or []:
                for ann in getattr(block, "annotations", []) or []:
                    if getattr(ann, "type", None) == "file_citation":
                        sources.append({
                            "filename": getattr(ann, "filename", "file"),
                            "page": getattr(ann, "page", None)
                        })
    except Exception:
        pass

    # de-dupe and append
    seen, uniq = set(), []
    for s in sources:
        key = (s["filename"], s.get("page"))
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    if uniq:
        text += "\n\nSources:\n"
        for i, s in enumerate(uniq, 1):
            page = f", p.{s['page']}" if s.get("page") else ""
            text += f"{i}. {s['filename']}{page}\n"
    return text


def _get_last_response_id():
    """Get the most recent assistant response ID from chat history."""
    chat = session.get("chat", [])
    for msg in reversed(chat):
        if msg["role"] == "assistant" and msg.get("response_id"):
            return msg["response_id"]
    return None


# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def home():
    chat = _ensure_history()
    source = _ensure_source()

    # Convert stored ISO times to friendly strings for display
    messages = []
    for m in chat:
        messages.append({
            "role": m["role"],
            "text": m["text"],
            "time": _iso_to_local(m.get("time", "")),
        })

    return render_template("template.html", messages=messages, source=source)


@app.route("/toggle_source", methods=["POST"])
def toggle_source():
    """Toggle between BRD and FSD vector stores."""
    current = _ensure_source()
    session["source"] = "fsd" if current == "brd" else "brd"
    session.modified = True
    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    q = (request.form.get("q") or "").strip()
    if not q:
        return redirect(url_for("home"))

    chat = _ensure_history()
    source = _ensure_source()

    # 1) Add the user message
    chat.append({"role": "user", "text": q, "time": _now_iso()})
    session.modified = True

    # 2) Select the appropriate vector store based on current source
    vector_store_id = VECTOR_STORE_BRD if source == "brd" else VECTOR_STORE_FSD

    # 3) Get the previous response ID for multi-turn conversation
    previous_response_id = _get_last_response_id()

    # 4) Build the input for the current turn
    # If this is the first message, include system prompt with the user message
    if previous_response_id is None:
        input_content = [
            {"role": "system", "content": _get_system_prompt(source)},
            {"role": "user", "content": q}
        ]
    else:
        # For follow-up messages, just send the user's question
        # The previous context is maintained via previous_response_id
        input_content = q

    # 5) Build kwargs
    kwargs = {
        "model": MODEL,
        "input": input_content,
        "store": True  # Required to use previous_response_id in future calls
    }

    # Add previous_response_id if this is a follow-up
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    has_store = bool(vector_store_id)

    if has_store:
        try:
            # modern style (new SDK)
            resp = client.responses.create(
                **kwargs,
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            )
        except TypeError:
            # fallback for older SDKs
            resp = client.responses.create(
                **kwargs,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                }],
            )
    else:
        # No vector store configured: still answer, but likely out-of-scope
        resp = client.responses.create(**kwargs)

    # 6) Extract text + citations
    assistant_text = _extract_text_and_sources(resp)
    if has_store and "Sources:" not in assistant_text:
        assistant_text = "This question appears to be outside the scope of the BRDs and FSDs."

    if not assistant_text.strip():
        assistant_text = "I didn't receive any text in the response."

    # 7) Append assistant reply to chat with response ID
    chat.append({
        "role": "assistant",
        "text": assistant_text,
        "time": _now_iso(),
        "response_id": resp.id  # Store the response ID for future multi-turn
    })
    session.modified = True

    return redirect(url_for("home"))


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat", None)
    # Keep the source selection when resetting chat
    return redirect(url_for("home"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
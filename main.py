import os
import time
from datetime import datetime, timezone,timedelta
from flask import Flask, request, session, render_template, redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

client = OpenAI()  # Uses OPENAI_API_KEY
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID", "").strip()

# ------------- Helpers -------------

def _ensure_history():
    """Initialize a simple in-session chat history."""
    if "chat" not in session:
        session["chat"] = []  # list of dicts: {"role": "user"/"assistant", "text": "...", "time": "..."}
    return session["chat"]

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _iso_to_local(ts: str) -> str:
    """
    Convert ISO timestamp (UTC) to Qatar local time (UTC+3)
    and render it as a readable local time label.
    """
    try:
        # Parse the incoming ISO string (replace Z → UTC)
        dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # Convert to Qatar timezone (UTC+3)
        qatar_tz = timezone(timedelta(hours=3))
        dt_qatar = dt_utc.astimezone(qatar_tz)
        # Format nicely
        return dt_qatar.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""
    


BRD_FSD_SYSTEM = """
You are an AI assistant specialized in answering questions only from official BRDs (Arabic) and FSDs (English)...
Role:
You are an AI assistant specialized in answering questions only from official Business Requirements Documents (BRDs) and Functional Specification Documents (FSDs).

Your core purpose is to help users understand system workflows, processes, business rules, and requirements extracted from those documents.
You must not answer questions outside the BRDs/FSDs domain.

All factual answers must come exclusively from the indexed BRD (Arabic) and FSD (English) documents.

Always reply in the same language as the user’s question.

Provide a clear, structured answer.

Always include citations from the retrieved documents  (if available) — specify:

Document name

Section name or heading

Page number


Be concise but faithful — summarize multiple chunks logically without changing their meaning.

Never guess or assume. If no direct match is found, say clearly:

“This question appears to be outside the scope of the BRDs and FSDs.”

Support follow-up questions — retain context across turns within the session.

If multiple related sections are found, merge insights and list all relevant sources.

Maintain a professional, explanatory tone suitable for analysts or technical users.


If the query concerns anything outside the BRD/FSD domain (e.g., religious, legal, personal, or unrelated technical questions):

“I can only answer questions that are documented in the BRDs and FSDs.”

If unsure, use:

“This question appears to be outside the scope of the BRDs and FSDs.”


Preserve the chat context to understand follow-up questions.

If the user asks a follow-up like “What happens next?” — infer based on previous question context and retrieved document continuity.


If user explicitly mentions “from BRDs” or “from FSDs”, obey that directive.

If both exist and overlap, merge results from both indexes but tag each source clearly.
"""

def _to_responses_messages(chat):
    msgs = [{"role": "system", "content": BRD_FSD_SYSTEM}]
    for m in chat:
        msgs.append({"role": m["role"], "content": m["text"]})
    return msgs


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
            seen.add(key); uniq.append(s)
    if uniq:
        text += "\n\nSources:\n"
        for i, s in enumerate(uniq, 1):
            page = f", p.{s['page']}" if s.get("page") else ""
            text += f"{i}. {s['filename']}{page}\n"
    return text

# ------------- Routes -------------

@app.route("/", methods=["GET"])
def home():
    chat = _ensure_history()
    # Convert stored ISO times to friendly strings for display
    messages = []
    for m in chat:
        messages.append({
            "role": m["role"],
            "text": m["text"],
            "time": _iso_to_local(m.get("time", "")),
        })
    return render_template("template.html", messages=messages)

@app.route("/ask", methods=["POST"])
def ask():
    q = (request.form.get("q") or "").strip()
    if not q:
        return redirect(url_for("home"))

    chat = _ensure_history()

    # 1) Add the user message
    chat.append({"role": "user", "text": q, "time": _now_iso()})
    session.modified = True

    # 2) Build messages with system policy
    input_messages = _to_responses_messages(chat)

    # 3) Build kwargs (robust to empty VECTOR_STORE_ID + old SDKs)
    kwargs = {"model": MODEL, "input": input_messages}

    has_store = bool(VECTOR_STORE_ID)
    if has_store:
        try:
            # modern style
            resp = client.responses.create(
                **kwargs,
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
            )
        except TypeError:
            # fallback for older SDKs
            resp = client.responses.create(
                **kwargs,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID],
                }],
            )
    else:
        # No vector store configured: still answer, but likely out-of-scope
        resp = client.responses.create(**kwargs)

    # 4) Extract text + citations (your helper)
    assistant_text = _extract_text_and_sources(resp)
    if has_store and "Sources:" not in assistant_text:
        # Be strict only when a store exists
        assistant_text = "This question appears to be outside the scope of the BRDs and FSDs."

    if not assistant_text.strip():
        assistant_text = "I didn't receive any text in the response."

    # 5) Append assistant reply to chat
    chat.append({"role": "assistant", "text": assistant_text, "time": _now_iso()})
    session.modified = True

    return redirect(url_for("home"))


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY and VECTOR_STORE_ID (optional but recommended) are set
    app.run(debug=True)

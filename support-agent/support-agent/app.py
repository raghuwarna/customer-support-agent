"""Customer support agent: Gemini + tool calling + Gradio UI."""
import os, time
import gradio as gr
from google import genai
from google.genai import types
import tools

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise SystemExit("Set the GEMINI_API_KEY environment variable.")
client = genai.Client(api_key=API_KEY)

# Free-tier quotas are per model, so we fall through this list when one runs out for the day.
MODELS = [m.strip() for m in os.environ.get(
    "MODELS", "gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash").split(",")]
model_idx = 0
MAX_USER_TURNS = 25      # per visitor session, protects the free quota
MAX_TOOL_ROUNDS = 8

SYSTEM = """You are a customer support agent for an online electronics store.
Rules:
- Before sharing any order or ticket information, verify the customer with verify_customer (customer ID + email).
- Answer ONLY from tool results. If you don't have the information, say so. Never invent order details, dates or policies.
- Never reveal or discuss any other customer's data. If a tool says an order or ticket is not found, say you couldn't find it for their account.
- Cancellation and refund tools only CHECK eligibility. Never claim a cancellation or refund has been processed; say what the policy allows and what happens next.
- Answer the customer's actual question first. If they ask about "my order" without an ID, call list_orders, summarize the orders briefly (ID, status, total) and ask which one they mean. Do not call get_order for every order.
- Call escalate_to_human only when it is relevant to what the customer is asking: a check says needs_human, the order they asked about has data_conflicts, they ask for a human, they are angry, or it is outside your scope. Do not escalate just because some other order looks odd.
- Be concise, polite and clear. Amounts are in Indian rupees (Rs.)."""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[types.Tool(function_declarations=tools.DECLARATIONS)],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

def call_model(contents):
    global model_idx
    for attempt in range(8):
        name = MODELS[model_idx]
        try:
            return client.models.generate_content(model=name, contents=contents, config=config)
        except Exception as e:
            msg = str(e)
            if "429" in msg and "PerDay" in msg and model_idx < len(MODELS) - 1:
                print(f"[quota] daily limit reached on {name}, switching model", flush=True)
                model_idx += 1
                continue
            if "429" in msg or "503" in msg:
                time.sleep(4 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Model unavailable after retries")

def chat_turn(state, user_text):
    contents = state["contents"]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    for _ in range(MAX_TOOL_ROUNDS):
        resp = call_model(contents)
        content = resp.candidates[0].content
        contents.append(content)
        parts = content.parts or []
        calls = [p.function_call for p in parts if p.function_call]
        if not calls:
            return "".join(p.text for p in parts if p.text) or "Sorry, I couldn't produce an answer."
        results = []
        for fc in calls:
            args = dict(fc.args or {})
            out = tools.run_tool(fc.name, args, state)
            print(f"[tool] {fc.name}({args})", flush=True)
            results.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=results))
    return "I'm having trouble completing this. Let me hand you to a human agent."

sessions = {}   # one entry per visitor: chat history + verified customer

def respond(message, history, request: gr.Request):
    state = sessions.setdefault(request.session_hash,
                                {"contents": [], "customer_id": None, "turns": 0})
    state["turns"] += 1
    if state["turns"] > MAX_USER_TURNS:
        return "This demo session has reached its message limit. Please refresh the page to start again."
    try:
        return chat_turn(state, message)
    except Exception as e:
        print("[error]", repr(e)[:300], flush=True)
        return "Sorry, the assistant is temporarily unavailable. Please try again in a moment."

demo = gr.ChatInterface(
    respond,
    title="Customer Support Agent",
    description="Demo login: CUST0007 / user007@example.com. Ask about your orders, cancellations or refunds.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

# AI Customer Support Agent

A tool-calling support agent (Gemini + Gradio) over a relational e-commerce dataset
(customers, products, orders, order_items, support_tickets).

## How it works
- `tools.py`: SQLite database (rebuilt in memory from `data/*.csv` on start), policy rules, and the tools
  the model can call: verify_customer, list_orders, get_order, get_ticket, check_cancellation,
  check_refund, escalate_to_human.
- `app.py`: the agent loop (model asks for a tool, code runs it, result goes back) and the Gradio chat UI.

## Guardrails
- Identity is verified (customer ID + email) before any data is returned.
- The model never chooses the customer ID: the code injects the verified one, so another
  customer's orders and tickets cannot be read, even if the model is manipulated.
- "Not found" is returned for both missing and not-owned records, so existence isn't leaked.
- Tools are read-only and parameterized (no model-written SQL). Policy decisions are made in code, not in the prompt.
- Conflicting data and over-limit refunds are escalated to a human.

## Data issues found
- All 100 support tickets reference an order that belongs to a different customer than the ticket's customer.
- Order status and payment status often contradict each other (e.g. Delivered with Failed payment).
- README says 40 products and 700+ order items; the CSVs contain 20 products and 569 order items.
- Product categories are inconsistent (e.g. earbuds under "Home"); every product is returnable.
- Ticket descriptions are 8 repeated templates.
- Many orders are stuck in early statuses months past their expected delivery date.

## Policy assumptions (edit in `tools.py`)
- Cancel only if the order is Confirmed/Processing and flagged cancellable.
- Return window: 7 days after the expected delivery date (no actual delivery date exists in the data).
- Auto-refund limit by tier: standard Rs.5,000, premium Rs.7,500, vip Rs.10,000; above that goes to a human.
- "Today" is fixed at 2026-10-07 for reproducibility.

## Run locally
    pip install -r requirements.txt
    set GEMINI_API_KEY=your_key      (Windows)   /   export GEMINI_API_KEY=your_key   (Mac/Linux)
    python app.py

## Deploy on Render (free web service)
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Environment variable: `GEMINI_API_KEY` (never commit the key)

## Limitations
- In-memory sessions and database: restarts reset everything (escalations are also printed to the logs).
- Verification by ID + email only; no real authentication or rate limiting beyond a per-session message cap.
- Free-tier model quotas are small; the app rotates through several Gemini models when one runs out.

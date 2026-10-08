"""Database, policy rules and tools for the support agent (no LLM code here)."""
import sqlite3, threading
from datetime import date
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TABLES = ["customers", "products", "orders", "order_items", "support_tickets"]

# ---- Policy (edit these; document them in the README) ----
AS_OF = date(2026, 10, 7)            # "today" for this dataset
RETURN_WINDOW_DAYS = 7               # days after delivery (expected_delivery is used as a stand-in)
CANCELLABLE_STATUSES = {"Confirmed", "Processing"}
AUTO_REFUND_LIMIT = {"standard": 5000, "premium": 7500, "vip": 10000}   # INR

# ---- In-memory SQLite, rebuilt from the CSVs on every start ----
_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()

def load_data():
    with _lock:
        for t in TABLES:
            pd.read_csv(DATA_DIR / f"{t}.csv").to_sql(t, _conn, if_exists="replace", index=False)
        _conn.execute("""CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, reason TEXT, created_at TEXT)""")
        _conn.commit()

def q(sql, params=()):
    with _lock:
        return [dict(r) for r in _conn.execute(sql, params).fetchall()]

load_data()

NOT_FOUND = {"error": "Order not found for this customer."}

def verify_customer(customer_id, email):
    rows = q("SELECT name, customer_tier, email FROM customers WHERE customer_id=?", (customer_id.strip(),))
    if rows and rows[0]["email"].lower() == email.strip().lower():
        return {"verified": True, "name": rows[0]["name"], "tier": rows[0]["customer_tier"]}
    return {"verified": False}

def _owned_order(order_id, customer_id):
    rows = q("SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id.strip(), customer_id))
    return rows[0] if rows else None

def _anomalies(o):
    s, p, found = o["status"], o["payment_status"], []
    if s == "Delivered" and p in ("Failed", "Pending"):
        found.append("Order is Delivered but payment is " + p)
    if s in ("Shipped", "Out for Delivery") and p == "Failed":
        found.append("Order is " + s + " but payment Failed")
    if s == "Cancelled" and p == "Paid":
        found.append("Order is Cancelled but still marked Paid (refund may be owed)")
    if s == "Refunded" and p != "Refunded":
        found.append("Order status is Refunded but payment status is " + p)
    if p == "Refunded" and s not in ("Cancelled", "Refunded", "Return Requested", "Delivered"):
        found.append("Payment is Refunded but order status is " + s)
    return found

def list_orders(customer_id):
    return q("""SELECT order_id, status, payment_status, total_amount, order_date
                FROM orders WHERE customer_id=? ORDER BY order_date DESC""", (customer_id,))

def get_order(order_id, customer_id):
    o = _owned_order(order_id, customer_id)
    if not o:
        return NOT_FOUND
    items = q("""SELECT p.product_name, i.quantity, i.unit_price, p.returnable, p.warranty_months
                 FROM order_items i JOIN products p ON p.product_id=i.product_id
                 WHERE i.order_id=?""", (o["order_id"],))
    return {**o, "items": items, "data_conflicts": _anomalies(o)}

def get_ticket(ticket_id, customer_id):
    rows = q("SELECT * FROM support_tickets WHERE ticket_id=? AND customer_id=?", (ticket_id.strip(), customer_id))
    if not rows:
        return {"error": "Ticket not found for this customer."}
    t = rows[0]
    # Dataset trap: a ticket's linked order often belongs to a different customer.
    if not _owned_order(t["order_id"], customer_id):
        t["order_id"] = None
        t["note"] = "Linked order does not belong to this customer; details withheld. Escalate."
    return t

def check_cancellation(order_id, customer_id):
    o = _owned_order(order_id, customer_id)
    if not o:
        return NOT_FOUND
    if o["status"] == "Cancelled":
        return {"decision": "not_applicable", "reason": "Order is already cancelled."}
    if o["cancellable"] and o["status"] in CANCELLABLE_STATUSES:
        return {"decision": "allowed", "reason": "Order has not shipped yet."}
    return {"decision": "not_allowed", "reason": f"Order is {o['status']} and cannot be cancelled."}

def check_refund(order_id, customer_id):
    o = _owned_order(order_id, customer_id)
    if not o:
        return NOT_FOUND
    s, p, amt = o["status"], o["payment_status"], o["total_amount"]
    tier = q("SELECT customer_tier FROM customers WHERE customer_id=?", (customer_id,))[0]["customer_tier"]
    conflicts = _anomalies(o)
    if conflicts:
        return {"decision": "needs_human", "reason": "Conflicting order/payment data: " + "; ".join(conflicts)}
    if s == "Refunded" or p == "Refunded":
        return {"decision": "not_eligible", "reason": "Already refunded."}
    if p != "Paid":
        return {"decision": "not_eligible", "reason": f"Payment is {p}, so nothing was charged."}
    if s in ("Cancelled", "Return Requested"):
        pass
    elif s == "Delivered":
        days = (AS_OF - date.fromisoformat(o["expected_delivery"])).days
        returnable = all(i["returnable"] for i in q(
            "SELECT p.returnable FROM order_items i JOIN products p ON p.product_id=i.product_id WHERE i.order_id=?",
            (o["order_id"],)))
        if days > RETURN_WINDOW_DAYS or not returnable:
            return {"decision": "not_eligible",
                    "reason": f"Outside the {RETURN_WINDOW_DAYS}-day return window or item not returnable."}
    else:
        return {"decision": "not_eligible", "reason": f"Order is {s}; refunds apply after delivery or cancellation."}
    if amt > AUTO_REFUND_LIMIT[tier]:
        return {"decision": "needs_human",
                "reason": f"Amount Rs.{amt} exceeds the {tier} auto-refund limit.", "amount": amt}
    return {"decision": "eligible", "reason": "Meets refund policy.", "amount": amt}

def escalate_to_human(customer_id, reason):
    with _lock:
        _conn.execute("INSERT INTO escalations (customer_id, reason, created_at) VALUES (?,?,?)",
                      (customer_id, reason, str(AS_OF)))
        _conn.commit()
    print(f"[ESCALATION] {customer_id}: {reason}", flush=True)   # visible in Render logs
    return {"status": "escalated", "message": "A human agent will follow up."}

# ---- Tool descriptions for the model ----
def _decl(name, desc, props=None):
    d = {"name": name, "description": desc}
    if props:
        d["parameters"] = {"type": "object",
                           "properties": {k: {"type": "string"} for k in props},
                           "required": list(props)}
    return d

DECLARATIONS = [
    _decl("verify_customer", "Verify identity. Must succeed before any other tool.", ["customer_id", "email"]),
    _decl("list_orders", "List the verified customer's orders."),
    _decl("get_order", "Get details of one of the customer's orders, including items and any data_conflicts.", ["order_id"]),
    _decl("get_ticket", "Get a support ticket belonging to the customer.", ["ticket_id"]),
    _decl("check_cancellation", "Check whether an order can be cancelled under policy.", ["order_id"]),
    _decl("check_refund", "Check refund eligibility under policy.", ["order_id"]),
    _decl("escalate_to_human", "Hand off to a human agent with a clear reason.", ["reason"]),
]

def run_tool(name, args, state):
    """The model never chooses customer_id: it comes from the verified session `state`."""
    if name == "verify_customer":
        r = verify_customer(str(args.get("customer_id", "")), str(args.get("email", "")))
        if r["verified"]:
            state["customer_id"] = args["customer_id"].strip()
        return r
    cid = state.get("customer_id")
    if not cid:
        return {"error": "Customer not verified yet. Ask for their customer ID and email."}
    try:
        if name == "list_orders":        return list_orders(cid)
        if name == "get_order":          return get_order(str(args["order_id"]), cid)
        if name == "get_ticket":         return get_ticket(str(args["ticket_id"]), cid)
        if name == "check_cancellation": return check_cancellation(str(args["order_id"]), cid)
        if name == "check_refund":       return check_refund(str(args["order_id"]), cid)
        if name == "escalate_to_human":  return escalate_to_human(cid, str(args["reason"]))
    except KeyError as e:
        return {"error": f"Missing argument: {e}"}
    return {"error": "Unknown tool"}

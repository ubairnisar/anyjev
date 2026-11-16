"""Define your own task in Python and judge text (works with text-only models too)."""
from anyjev import Judge, Task

task = Task(
    name="ticket-router",
    question="Which team should handle this support ticket?",
    options={
        "billing": "Billing - payments, invoices, refunds",
        "bug": "Engineering - something is broken",
        "account": "Accounts - login, password, profile",
        "other": "Anything else",
    },
)

judge = Judge()  # ANYJEV_BASE_URL / ANYJEV_MODEL or localhost:8000
r = judge.classify(task, text="I was charged twice for my subscription this month.")
print(r.label, {k: round(v, 3) for k, v in r.probs.items()})

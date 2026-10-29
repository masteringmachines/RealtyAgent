#!/usr/bin/env python3
"""realty_agent - lightweight conversational intake for real estate teams.

Instead of a static contact form, Claude interviews the buyer / borrower / seller /
tenant in plain language, records structured fields as it learns them, then writes a
scored lead record plus a hand-off brief for the human agent.

Templates (inspired by the "Concierge / Interviewer / Evaluator" agent patterns):
  buyer     Qualify home buyers (budget, beds, areas, timeline, pre-approval)
  prequal   Prequalify mortgage borrowers (+ indicative affordability estimate)
  lead      Real-estate lead capture (buy / sell / both)
  debrief   Post-tour buyer debrief
  tenant    Tenant satisfaction survey (flags churn risk)

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python realty_agent.py templates
  python realty_agent.py run buyer            # interactive chat in the terminal
  python realty_agent.py leads                # ranked table of saved records
  python realty_agent.py show <file.json>     # print a saved hand-off brief

Type /done to end early (partial record is still saved). Records go to ./leads/.
Only dependency: `pip install anthropic`.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
LEADS_DIR = Path(os.getenv("LEADS_DIR", "./leads"))
MAX_TURNS = 30


def F(desc, required=False):
    return {"desc": desc, "required": required}


# --------------------------------------------------------------------------- templates
TEMPLATES = {
    "buyer": {
        "title": "Qualify Home Buyers",
        "goal": "Understand what this person wants in their next home and how ready they are to buy, so an agent can follow up with context.",
        "opening": "Start by asking what matters most to them in their next home.",
        "fields": {
            "intent": F("buy, sell, or both", True),
            "budget_max": F("maximum budget in USD (number)", True),
            "bedrooms": F("minimum bedrooms (number)"),
            "locations": F("neighborhoods/areas/school districts they would consider (list of strings)", True),
            "must_haves": F("non-negotiables, in their own words (list of strings)"),
            "timeline_months": F("months until they need to move (number)", True),
            "preapproved": F("true/false: do they have a mortgage pre-approval", True),
            "household": F("who is moving (e.g. family of 4, two dogs)"),
            "motivation": F("why they are moving, in their own words"),
        },
    },
    "prequal": {
        "title": "Prequalify Mortgage Borrowers",
        "goal": "Collect the basics a loan officer needs for a first conversation. You are NOT a lender and must not promise approval or rates.",
        "opening": "Start by asking whether they are purchasing or refinancing.",
        "fields": {
            "purpose": F("purchase or refinance", True),
            "property_price": F("target home price in USD (number)", True),
            "annual_income": F("combined gross annual income in USD (number)", True),
            "monthly_debts": F("total monthly debt payments in USD, excluding rent/mortgage (number)", True),
            "down_payment": F("cash available for down payment in USD (number)", True),
            "credit_range": F("self-reported credit range: <620, 620-679, 680-739, 740+"),
            "timeline_months": F("months until they want to close (number)"),
        },
    },
    "lead": {
        "title": "Real Estate Lead Capture",
        "goal": "Replace a contact form: learn whether they want to buy, sell, or both, their timing, and the situation behind it.",
        "opening": "Start by asking whether they are looking to buy, sell, or both.",
        "fields": {
            "intent": F("buy, sell, or both", True),
            "timeline_months": F("months until they need to move (number)", True),
            "situation": F("the circumstances driving the move, in their words", True),
            "property_address": F("address of the home they want to sell, if any"),
            "price_expectation": F("what they hope to sell/buy for in USD (number)"),
            "contact_preference": F("preferred contact method and time"),
        },
    },
    "debrief": {
        "title": "Post-Tour Buyer Debrief",
        "goal": "Capture honest, unstructured feedback after a showing, including gut feelings they cannot fully explain.",
        "opening": "Start by asking which home they just toured and their first impression.",
        "fields": {
            "property": F("address or description of the home toured", True),
            "liked": F("what they liked (list of strings)", True),
            "disliked": F("what they disliked (list of strings)", True),
            "price_opinion": F("too high / fair / good value, with reason"),
            "offer_likelihood": F("0-10 likelihood they will make an offer (number)", True),
            "blockers": F("anything in the way of an offer (list of strings)"),
        },
    },
    "tenant": {
        "title": "Tenant Satisfaction Survey",
        "goal": "Find out how the tenant experiences management, especially maintenance responsiveness, and whether they will renew.",
        "opening": "Start by asking how responsive management has been to their maintenance requests.",
        "fields": {
            "maintenance_responsiveness": F("their experience, in their words", True),
            "open_issue": F("any unresolved issue and how long it has been open"),
            "satisfaction": F("overall satisfaction 0-10 (number)", True),
            "would_renew": F("true/false: would they renew their lease", True),
            "suggestions": F("what would improve things"),
        },
    },
}


# --------------------------------------------------------------------------- scoring
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def affordability(f, rate=0.065, years=30):
    """Indicative only: 28% front-end / 36% back-end rules, P&I ~80% of the PITI budget."""
    inc = _num(f.get("annual_income"))
    if not inc:
        return None
    monthly = inc / 12
    debts = _num(f.get("monthly_debts")) or 0
    piti = max(min(0.28 * monthly, 0.36 * monthly - debts), 0)
    r, n = rate / 12, years * 12
    loan = piti * 0.8 * (1 - (1 + r) ** -n) / r
    down = _num(f.get("down_payment")) or 0
    max_price = loan + down
    out = {
        "assumptions": f"{rate:.1%} rate, {years}y fixed, 28/36 DTI rules, taxes+insurance ~20% of payment",
        "max_monthly_housing_payment": round(piti),
        "indicative_max_loan": round(loan),
        "indicative_max_price": round(max_price),
    }
    price = _num(f.get("property_price"))
    if price:
        out["target_vs_indicative"] = (
            "within range" if price <= max_price else
            "stretch" if price <= max_price * 1.15 else "above range"
        )
    return out


def score(tpl, f):
    """Returns (score 0-100 | None, tier, flags, derived)."""
    flags, derived, s = [], {}, None
    if tpl in ("buyer", "lead"):
        s = 0
        t = _num(f.get("timeline_months"))
        if t is not None:
            s += 35 if t <= 3 else 20 if t <= 6 else 8
        if f.get("preapproved") is True:
            s += 30
        elif f.get("preapproved") is False:
            flags.append("Not pre-approved: consider a lender intro")
        if _num(f.get("budget_max")) or _num(f.get("price_expectation")):
            s += 15
        if f.get("locations") or f.get("property_address"):
            s += 10
        if f.get("motivation") or f.get("situation"):
            s += 10
        if f.get("intent") == "both":
            flags.append("Buy + sell: potential double transaction")
    elif tpl == "prequal":
        derived["affordability"] = affordability(f)
        a = derived["affordability"]
        s = 0
        if a:
            s += {"within range": 50, "stretch": 30, "above range": 10}.get(a.get("target_vs_indicative"), 25)
            if a.get("target_vs_indicative") == "above range":
                flags.append("Target price above indicative range")
        t = _num(f.get("timeline_months"))
        if t is not None:
            s += 30 if t <= 3 else 15
        s += 20 if (_num(f.get("down_payment")) or 0) > 0 else 0
    elif tpl == "debrief":
        o = _num(f.get("offer_likelihood"))
        s = round(o * 10) if o is not None else None
        if f.get("blockers"):
            flags.append("Blockers to address before an offer")
    elif tpl == "tenant":
        sat = _num(f.get("satisfaction"))
        s = round(sat * 10) if sat is not None else None
        if f.get("would_renew") is False or (sat is not None and sat <= 4):
            flags.append("CHURN RISK: follow up personally")
        if f.get("open_issue"):
            flags.append("Unresolved maintenance issue")
    s = min(s, 100) if s is not None else None
    tier = None if s is None else "hot" if s >= 70 else "warm" if s >= 40 else "cold"
    return s, tier, flags, derived


# --------------------------------------------------------------------------- agent
def system_prompt(key):
    t = TEMPLATES[key]
    fields = "\n".join(
        f"- {k}{' (REQUIRED)' if v['required'] else ''}: {v['desc']}" for k, v in t["fields"].items()
    )
    return f"""You are a friendly intake assistant for a real estate brokerage, running "{t['title']}".
Goal: {t['goal']}

Fields to learn:
{fields}

Rules:
- Ask ONE short question at a time, conversationally. Follow up on interesting details; never read the list like a form.
- When the person mentions several facts in one answer, record all of them.
- Call record_fields as soon as you learn something, in the same turn as your next question. Never invent or assume values; leave unknown fields out. People may decline any question.
- When every REQUIRED field is recorded (or the person wants to stop), thank them and call finish_interview with a 2-3 sentence summary written for the human agent who will follow up. If it reports missing fields, keep going.
- You are not a lender, lawyer, or appraiser: no approval promises, rates, legal or investment advice.
- Never ask for SSNs, bank/card numbers or passwords. If offered, tell them not to share it.
- {t['opening']}"""


TOOLS = [
    {
        "name": "record_fields",
        "description": "Record structured facts learned from the user's latest message.",
        "input_schema": {
            "type": "object",
            "properties": {"fields": {"type": "object", "description": "field name -> value"}},
            "required": ["fields"],
        },
    },
    {
        "name": "finish_interview",
        "description": "End the interview once all required fields are recorded.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]

SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def redact(text):
    clean = CARD.sub("[REDACTED]", SSN.sub("[REDACTED]", text))
    return clean, clean != text


class Interview:
    def __init__(self, key, client):
        self.key, self.client = key, client
        self.fields, self.summary, self.done = {}, None, False
        self.transcript = []
        self.messages = []

    def _missing(self):
        return [k for k, v in TEMPLATES[self.key]["fields"].items() if v["required"] and k not in self.fields]

    def _call_tool(self, name, args):
        if name == "record_fields":
            known = TEMPLATES[self.key]["fields"]
            new = {k: v for k, v in (args.get("fields") or {}).items() if k in known and v not in (None, "", [])}
            self.fields.update(new)
            return f"Recorded {sorted(new)}. Still missing required: {self._missing() or 'none'}"
        if name == "finish_interview":
            missing = self._missing()
            if missing:
                return f"Cannot finish yet. Missing required fields: {missing}"
            self.summary, self.done = args.get("summary", ""), True
            return "Interview closed."
        return f"Unknown tool {name}"

    def step(self, user_text):
        """Send one user message (or None to open) and return the assistant's reply text."""
        self.messages.append({"role": "user", "content": user_text or "Begin the conversation."})
        if user_text:
            self.transcript.append({"role": "user", "text": user_text})
        said = []
        for _ in range(6):
            resp = self.client.messages.create(
                model=MODEL, max_tokens=700, system=system_prompt(self.key),
                tools=TOOLS, messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": resp.content})
            said += [b.text for b in resp.content if b.type == "text" and b.text.strip()]
            if resp.stop_reason != "tool_use":
                break
            results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": self._call_tool(b.name, b.input)}
                for b in resp.content if b.type == "tool_use"
            ]
            self.messages.append({"role": "user", "content": results})
        reply = "\n".join(said).strip()
        if reply:
            self.transcript.append({"role": "assistant", "text": reply})
        return reply

    def record(self):
        s, tier, flags, derived = score(self.key, self.fields)
        return {
            "template": self.key,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "complete": self.done,
            "missing_required": self._missing(),
            "fields": self.fields,
            "summary": self.summary or "(interview ended before the assistant could summarise)",
            "score": s, "tier": tier, "flags": flags, "derived": derived,
            "transcript": self.transcript,
        }


def brief(rec):
    t = TEMPLATES[rec["template"]]["title"]
    lines = [f"=== {t} | {rec['created']} ==="]
    if rec["score"] is not None:
        lines.append(f"Score: {rec['score']}/100 ({rec['tier'].upper()})")
    lines.append(f"Summary: {rec['summary']}")
    lines.append("\nCaptured:")
    for k, v in rec["fields"].items():
        lines.append(f"  - {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
    a = rec["derived"].get("affordability")
    if a:
        lines.append("\nIndicative affordability (NOT a lender decision):")
        lines += [f"  - {k}: {v:,}" if isinstance(v, int) else f"  - {k}: {v}" for k, v in a.items()]
    if rec["missing_required"]:
        lines.append(f"\nStill missing: {', '.join(rec['missing_required'])}")
    for fl in rec["flags"]:
        lines.append(f"  ! {fl}")
    return "\n".join(lines)


def save(rec):
    LEADS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADS_DIR / f"{rec['template']}-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return path


def run_interview(key, client, ask=input, say=print):
    iv = Interview(key, client)
    say(f"\n[{TEMPLATES[key]['title']}]  (type /done to finish early)\n")
    say(f"Agent: {iv.step(None)}\n")
    for _ in range(MAX_TURNS):
        if iv.done:
            break
        try:
            text = ask("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("/done", "/quit"):
            break
        if not text:
            continue
        text, was_redacted = redact(text)
        if was_redacted:
            say("(Sensitive number removed before sending. Please don't share SSNs or card numbers.)")
        say(f"\nAgent: {iv.step(text)}\n")
    rec = iv.record()
    path = save(rec)
    say("\n" + brief(rec) + f"\n\nSaved: {path}")
    return rec


# --------------------------------------------------------------------------- CLI
def make_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    return anthropic.Anthropic()


def cmd_leads():
    rows = []
    for p in sorted(LEADS_DIR.glob("*.json")):
        r = json.loads(p.read_text())
        rows.append((r["score"] if r["score"] is not None else -1, p.name, r))
    if not rows:
        print("No records yet. Run: python realty_agent.py run buyer")
        return
    print(f"{'SCORE':>5}  {'TIER':<5} {'TEMPLATE':<8} {'FLAGS':<40} FILE")
    for s, name, r in sorted(rows, key=lambda x: -x[0]):
        print(f"{'-' if s < 0 else s:>5}  {(r['tier'] or '-'):<5} {r['template']:<8} {('; '.join(r['flags']))[:38]:<40} {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("templates")
    r = sub.add_parser("run")
    r.add_argument("template", choices=TEMPLATES)
    sub.add_parser("leads")
    s = sub.add_parser("show")
    s.add_argument("file")
    a = ap.parse_args()

    if a.cmd == "templates":
        for k, t in TEMPLATES.items():
            print(f"{k:<8} {t['title']}")
    elif a.cmd == "run":
        run_interview(a.template, make_client())
    elif a.cmd == "leads":
        cmd_leads()
    elif a.cmd == "show":
        p = Path(a.file)
        print(brief(json.loads((p if p.exists() else LEADS_DIR / p).read_text())))


if __name__ == "__main__":
    main()

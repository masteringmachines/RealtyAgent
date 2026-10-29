import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realty_agent as ra  # noqa: E402


def text(t):
    return NS(type="text", text=t)


def tool(i, n, a):
    return NS(type="tool_use", id=i, name=n, input=a)


def resp(blocks, stop):
    return NS(content=blocks, stop_reason=stop)


def fake_client(responses):
    it = iter(responses)

    class C:
        class messages:
            @staticmethod
            def create(**kw):
                return next(it)

    return C


def test_buyer_interview_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "LEADS_DIR", tmp_path)
    client = fake_client([
        resp([text("What matters most in your next home?")], "end_turn"),
        resp([tool("a", "record_fields", {"fields": {
            "intent": "buy", "bedrooms": 4, "budget_max": 650000,
            "locations": ["good schools"], "bogus": 1}}),
            text("Pre-approved? When do you need to move?")], "tool_use"),
        resp([text("Pre-approved? When do you need to move?")], "end_turn"),
        resp([tool("b", "record_fields", {"fields": {"preapproved": True, "timeline_months": 3}}),
              tool("c", "finish_interview", {"summary": "Ready buyer."})], "tool_use"),
        resp([text("Thanks! An agent will be in touch.")], "end_turn"),
    ])
    answers = iter(["4 bed near good schools, $650K", "pre-approved, by August, ssn 123-45-6789"])
    rec = ra.run_interview("buyer", client, ask=lambda p: next(answers), say=lambda *_: None)

    assert rec["complete"]
    assert rec["fields"]["budget_max"] == 650000
    assert "bogus" not in rec["fields"]          # unknown fields are dropped
    assert rec["tier"] == "hot"
    assert "123-45" not in str(rec["transcript"])  # SSN redacted before sending
    assert list(tmp_path.glob("buyer-*.json"))


def test_finish_rejected_until_required_fields_present():
    iv = ra.Interview("buyer", client=None)
    assert "Cannot finish" in iv._call_tool("finish_interview", {"summary": "x"})
    assert not iv.done


def test_prequal_affordability():
    s, tier, _, d = ra.score("prequal", {
        "annual_income": 120000, "monthly_debts": 500, "down_payment": 60000,
        "property_price": 425000, "timeline_months": 2})
    a = d["affordability"]
    assert a["max_monthly_housing_payment"] == 2800
    assert a["target_vs_indicative"] == "stretch"


def test_tenant_churn_flag():
    s, tier, flags, _ = ra.score("tenant", {"satisfaction": 3, "would_renew": False})
    assert tier == "cold" and any("CHURN" in f for f in flags)


def test_redact_card_and_ssn():
    out, changed = ra.redact("card 4111 1111 1111 1111 and 123-45-6789")
    assert changed and "4111" not in out and "123-45" not in out

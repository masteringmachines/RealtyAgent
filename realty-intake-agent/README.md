# Realty Intake Agent

A lightweight, single-file Python tool that replaces static real estate contact forms with
a short AI conversation. Claude interviews the buyer, borrower, seller or tenant, records
structured fields as it learns them, then saves a **scored lead record** and a **hand-off
brief** for the human agent.

## Templates

| Command | Use case | Extras |
|---|---|---|
| `run buyer` | Qualify home buyers | Hot/warm/cold score from timeline, pre-approval, budget |
| `run prequal` | Prequalify mortgage borrowers | Indicative affordability estimate (28/36 DTI) |
| `run lead` | Lead capture (buy / sell / both) | Flags buy+sell opportunities |
| `run debrief` | Post-tour buyer debrief | Offer-likelihood score, blockers |
| `run tenant` | Tenant satisfaction survey | Churn-risk and open-issue flags |

## Quick start

```bash
git clone <your-repo-url> && cd realty-intake-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python realty_agent.py run buyer      # interactive interview
python realty_agent.py leads          # ranked table of saved records
python realty_agent.py show <file>    # print a hand-off brief
```

Type `/done` to end an interview early; the partial record is still saved to `./leads/`.
Set `CLAUDE_MODEL` or `LEADS_DIR` to override the model or output folder.

## How it works

1. A per-template system prompt lists the fields to learn (required ones marked).
2. Claude asks one question at a time and calls a `record_fields` tool whenever it learns something.
3. `finish_interview` is rejected by the code until every required field is recorded.
4. Scoring and affordability math are deterministic Python, not LLM guesses, so they are easy to audit and tune (`score()` in `realty_agent.py`).

## Customising

Add a template by adding an entry to `TEMPLATES` (fields + goal + opening) and, optionally, a branch in `score()`.

## Privacy and compliance notes

- SSN-like and card-like numbers are stripped from user input before it is sent to the model.
- `leads/*.json` contains personal data and is git-ignored. Handle it according to your privacy obligations.
- The affordability figure is an indicative estimate using assumed rates, not a lender decision. The agent is instructed not to promise approval, rates, or give legal or investment advice.
- Anyone using this for lending or housing decisions should review fair-housing and lending regulations before deploying.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests use a scripted fake model, so no API key is needed.

## License

MIT

# Law Office AI Secretary

When nobody at the office answers, the call is forwarded to an AI assistant that talks with the caller
in natural Hebrew, takes a message, and emails it to the lawyer they asked for.

## How it works

```
office line, no answer after N rings ──forward──▶ Twilio number
  POST /voice/incoming   → TwiML <ConversationRelay> (Twilio does speech-to-text, TTS, barge-in)
  WS   /relay            → live conversation: the LLM streams each reply token by token
  call ends              → the LLM writes a structured summary → SQLite + email to the lawyer
```

The LLM is Gemini or Claude, chosen with `LLM_PROVIDER` in `.env` (`app/llm.py`).
**Gemini's free tier lets Google use the data you send**, so use it only for testing with
made-up details. Real client calls need a paid tier (Gemini paid or Claude).

- `app/agent/prompt.py`: who the assistant is and what the lawyer needs afterwards. It is short on
  purpose, with no script and no question order, so the model runs the conversation itself.
- `app/agent/conversation.py`: conversation history, streaming replies, interruptions, the `end_call` tool.
- `app/relay.py`: the ConversationRelay WebSocket protocol.
- `app/agent/summary.py`: transcript → structured message (name, number, lawyer, topic, urgency…).
- `app/notify.py`, `app/db.py`: email and the call log.
- `config.yaml`: office name, lawyers and emails, practice areas.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
cp .env.example .env          # fill in the keys
```

Get a Gemini key at https://aistudio.google.com/apikey and put it in `GEMINI_API_KEY`.
To switch to Claude later, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.

Edit `config.yaml` with the real office details.

### 1. Try it in the terminal (no phone needed)

```bash
.venv/Scripts/python -m scripts.chat
```

Same prompt and model as a real call. At the end it prints the email the lawyer would get.
This is the fastest way to tune how she talks.

### 2. A real phone call

1. Buy a Twilio number with voice. In the Twilio Console, accept the ConversationRelay /
   AI features terms, since ConversationRelay won't connect until they are accepted.
2. Run the server and expose it:
   ```bash
   .venv/Scripts/python -m uvicorn app.main:app --port 8000
   ngrok http 8000
   ```
   Put the ngrok https URL in `PUBLIC_BASE_URL` in `.env` and restart.
3. In the Twilio number's settings, set "A call comes in" → Webhook → `https://<ngrok>/voice/incoming` (POST).
4. Call the Twilio number and talk to her.
5. Try a few Hebrew voices (`TTS_PROVIDER` / `TTS_VOICE`) and keep the one that sounds most natural.

### 3. Go live

Ask the office's phone provider to change the "forward on no answer" target from the current answering
service to the Twilio number. Run them side by side for a while first.

## Running in the cloud (24/7)

Real calls need a server that is always on, not a laptop. The repo is ready for
[Render](https://render.com) (Docker + `render.yaml`), and the Dockerfile also works on Railway, Fly.io or Cloud Run.

1. Push this folder to a **private** GitHub repo (`.env` is git-ignored and never uploaded).
2. Render → New → Blueprint → choose the repo. It reads `render.yaml`, which sets an always-on plan,
   the Frankfurt region, a 1 GB disk for the call log at `/data`, and a health check.
3. Fill in the secrets it asks for (the same values as in your `.env`).
4. After the deploy, open `https://<your-app>.onrender.com/health`. It should say `"llm": "gemini"`.
5. In Twilio, set the number's "A call comes in" webhook to `https://<your-app>.onrender.com/voice/incoming`.
   `PUBLIC_BASE_URL` is filled in automatically on Render.

Before real clients call:
- Set `TWILIO_AUTH_TOKEN` so only Twilio can start conversations.
- Remove `EMAIL_TO_OVERRIDE` so messages go to the lawyers in `config.yaml`.
- Use a paid LLM tier (see the note at the top).

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

## Notes

- The opening line says she's a digital assistant. Callers should know they're talking to an AI.
- Transcripts are stored in `data/calls.db` and sent by email. This is client-confidential material,
  so keep the server and mailbox access tight.

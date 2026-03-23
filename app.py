from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import requests
import json

app = Flask(__name__)

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3.5:0.8b"

SYSTEM_PROMPT = """
You are Nordic Travel AI, a professional travel assistant for Finland and the Nordic region.

Identity rules:
- Never say you are Qwen, a language model, or an Ollama model.
- Always present yourself as Nordic Travel AI.
- If the user greets you, briefly introduce yourself as Nordic Travel AI.

Business rules:
- This MVP is package-based.
- You must only use the package facts provided in the prompt.
- You may explain, compare, recommend, summarize, and personalize the wording.
- You must NOT change package names, prices, day counts, route structure, destination countries, highlights, or optional upgrades.
- You must NOT say that you cannot generate travel itineraries or packages.
- If the user asks a package-related question, answer directly and confidently using the package information.
- If the user asks for a custom route, explain that the current MVP focuses on fixed packages, then suggest the closest package.

Writing style:
- Clear, natural, professional English
- Helpful and concise
- Medium-length answers
- Use markdown headings and bullet points when useful
""".strip()

PACKAGE_1 = """
Package 1 — Aurora Classic Finland
Origin market: Pakistan → Finland
Trip length: 10 days
Price: €5,000 / person
Flights: not included

Route structure:
- 3 days Helsinki
- 7 days Rovaniemi / Lapland

Highlights:
- Tuomiokirkko
- Suomenlinna
- Santa Claus Village
- multiple aurora nights
- snow activities bundle

Includes:
- accommodation
- key transfers
- core experiences
- basic customer support

Optional upgrades:
- private aurora chase
- igloo night
- halal dining plan
- eSIM
""".strip()

PACKAGE_2 = """
Package 2 — Nordic Aurora Deep
Origin market: China → Finland + Norway
Trip length: 14 days
Price: €6,000 / person
Flights: not included

Route structure:
- Helsinki
- Turku
- Rovaniemi
- Tromsø

Highlights:
- forest snow scenery
- coastal aurora setting
- richer photography opportunities
- richer experiences

Includes:
- accommodation
- key transfers
- core experiences
- basic customer support

Optional upgrades:
- private aurora chase
- igloo night
- dining support plan
- eSIM
""".strip()

PACKAGE_3 = """
Package 3 — Winter Starter
Trip length: 7 days
Price: €3,500 / person
Flights: not included

Route structure:
- Helsinki
- Lapland

Designed for:
- first-time Nordics travelers

Positioning:
- shorter loop
- fewer moves
- entry winter product
""".strip()

ALL_PACKAGES = f"""
{PACKAGE_1}

{PACKAGE_2}

{PACKAGE_3}
""".strip()


def normalize(text: str) -> str:
    return text.lower().strip()


def detect_intent(user_input: str):
    t = normalize(user_input)

    # greeting
    if t in {"hello", "hi", "hey", "hello!", "hi!", "hey!", "introduce yourself", "who are you"}:
        return "greeting", ""

    # explain / direct package match
    if "package 1" in t or "aurora classic" in t or "classic finland aurora" in t or ("pakistan" in t and "finland" in t):
        if "compare" not in t:
            return "explain_package_1", PACKAGE_1

    if "package 2" in t or "nordic aurora deep" in t or "finland and norway" in t or "finland + norway" in t or "tromsø" in t or "tromso" in t:
        if "compare" not in t:
            return "explain_package_2", PACKAGE_2

    if "package 3" in t or "winter starter" in t or "shorter and easier winter trip" in t or "fewer moves" in t:
        if "compare" not in t:
            return "explain_package_3", PACKAGE_3

    # compare
    if "compare" in t and "package 1" in t and "package 2" in t:
        return "compare_1_2", f"{PACKAGE_1}\n\n{PACKAGE_2}"

    if "compare" in t and "package 1" in t and "package 3" in t:
        return "compare_1_3", f"{PACKAGE_1}\n\n{PACKAGE_3}"

    if "compare" in t and "package 2" in t and "package 3" in t:
        return "compare_2_3", f"{PACKAGE_2}\n\n{PACKAGE_3}"

    # recommendation
    if "first-time" in t or "first time" in t or "first-time travelers" in t:
        return "recommend_first_time", ALL_PACKAGES

    if "family" in t and "china" in t:
        return "recommend_family_china", ALL_PACKAGES

    if "pakistan" in t:
        return "recommend_pakistan", ALL_PACKAGES

    if "classic finland aurora" in t:
        return "explain_package_1", PACKAGE_1

    if "finland and norway winter journey" in t or "deeper finland and norway" in t:
        return "explain_package_2", PACKAGE_2

    if "shorter and easier winter trip" in t:
        return "explain_package_3", PACKAGE_3

    # generic package/travel question
    package_keywords = [
        "package", "trip", "travel", "plan", "route", "aurora",
        "finland", "norway", "lapland", "helsinki", "tromsø", "tromso"
    ]
    if any(word in t for word in package_keywords):
        return "general_package_question", ALL_PACKAGES

    return "general_non_package", ""


def build_user_prompt(user_input: str, intent: str, facts: str) -> str:
    if intent == "greeting":
        return f"""
User message:
{user_input}

Task:
Reply briefly as Nordic Travel AI.
Introduce yourself as a package-based Nordic winter travel assistant.
Do not mention any package unless the user asks.
""".strip()

    if intent.startswith("explain_package_"):
        return f"""
User request:
{user_input}

Selected task:
Explain this package clearly and attractively.

Package facts:
{facts}

Rules:
- Keep all facts unchanged.
- Present the package as a real travel product.
- Include who it is suitable for.
- Use markdown headings and bullet points.
""".strip()

    if intent.startswith("compare_"):
        return f"""
User request:
{user_input}

Selected task:
Compare these packages clearly.

Package facts:
{facts}

Rules:
- Keep all facts unchanged.
- Compare route structure, duration, price, and travel style.
- End with a short recommendation on when each package fits better.
- Use markdown headings and bullet points.
""".strip()

    if intent == "recommend_first_time":
        return f"""
User request:
{user_input}

Selected task:
Recommend the best package for first-time Nordics travelers.

Available package facts:
{facts}

Rules:
- Keep all facts unchanged.
- Recommend one package first, then optionally mention one alternative.
- Explain the recommendation clearly.
- Use markdown headings and bullet points.
""".strip()

    if intent == "recommend_family_china":
        return f"""
User request:
{user_input}

Selected task:
Recommend the best package for a family from China.

Available package facts:
{facts}

Rules:
- Keep all facts unchanged.
- Recommend one package first, then optionally mention one alternative.
- Explain the recommendation clearly.
- Use markdown headings and bullet points.
""".strip()

    if intent == "recommend_pakistan":
        return f"""
User request:
{user_input}

Selected task:
Recommend the most relevant package for a traveler from Pakistan.

Available package facts:
{facts}

Rules:
- Keep all facts unchanged.
- Recommend the most relevant package first.
- Explain why it fits.
- Use markdown headings and bullet points.
""".strip()

    if intent == "general_package_question":
        return f"""
User request:
{user_input}

Selected task:
Answer this package-related question using the fixed package facts below.

Available package facts:
{facts}

Rules:
- Keep all facts unchanged.
- If the user asks for a recommendation, recommend the closest package.
- If the user asks for a custom route, say that the current MVP is package-based and suggest the closest package.
- Use markdown when helpful.
""".strip()

    return f"""
User message:
{user_input}

Task:
Reply briefly as Nordic Travel AI.
If the question is not package-related, explain that this MVP focuses on fixed Nordic winter travel packages.
Then suggest asking about the available travel packages.
""".strip()


def call_ollama_stream(user_prompt: str):
    payload = {
        "model": MODEL_NAME,
        "stream": True,
        "think": False,
        "keep_alive": "15m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": 0.2,
            "num_predict": 1024
        }
    }

    with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=180) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = obj.get("message", {})
            content = message.get("content", "")
            done = obj.get("done", False)

            if content:
                yield content

            if done:
                break


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "success": True,
        "message": "Flask backend is working."
    })


@app.route("/generate_stream", methods=["POST"])
def generate_stream():
    data = request.get_json(silent=True)
    if not data:
        return Response("Error: request body is empty.", mimetype="text/plain")

    user_input = str(data.get("input", "")).strip()
    if not user_input:
        return Response("Error: input is empty.", mimetype="text/plain")

    intent, facts = detect_intent(user_input)
    user_prompt = build_user_prompt(user_input, intent, facts)

    def generate_chunks():
        try:
            for chunk in call_ollama_stream(user_prompt):
                yield chunk
        except requests.exceptions.ConnectionError:
            yield "\n\n[Error: Cannot connect to Ollama. Make sure Ollama is running.]"
        except requests.exceptions.Timeout:
            yield "\n\n[Error: Ollama request timed out.]"
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

    return Response(stream_with_context(generate_chunks()), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)
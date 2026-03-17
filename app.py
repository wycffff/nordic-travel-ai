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
- If the user greets you, briefly introduce yourself as a Nordic travel planning assistant.

Response rules:
- Write in clear, natural English.
- Be practical, concise, and helpful.
- Do not generate a travel itinerary unless the user is clearly asking for travel planning.
- If the user asks for a travel route or itinerary, use the provided route template if available.
- Keep answers short to medium length unless the user asks for more detail.
""".strip()

ROUTE_3_DAY_HELSINKI = """
Route name: 3-Day Helsinki City Break

Best for:
- First-time visitors
- Couples or small families
- Short urban trip

Plan:
Day 1: Senate Square, Helsinki Cathedral, Market Square, Esplanadi, Finnish dinner
Day 2: Suomenlinna, Design District, local café, optional evening sauna
Day 3: Temppeliaukio Church, Oodi Library or museum, free time, optional Porvoo

Route logic:
A compact Helsinki introduction with culture, landmarks, and local atmosphere.
""".strip()

ROUTE_7_DAY_FINLAND = """
Route name: 7-Day Finland and Lapland Highlights

Best for:
- First-time Finland visitors
- Travelers who want both city and northern experience

Plan:
Day 1: Arrival in Helsinki
Day 2: Helsinki highlights
Day 3: Turku cultural visit
Day 4: Travel to Rovaniemi
Day 5: Santa Claus Village and Lapland highlights
Day 6: Nature or seasonal activities
Day 7: Return and departure

Route logic:
Balanced first Finland trip with Helsinki, Turku, and Lapland.
""".strip()

ROUTE_15_DAY_NORDIC = """
Route name: 15-Day Finland-Focused Nordic Discovery Journey

Best for:
- Long-haul travelers from Asia
- Families or couples
- People who want a deeper trip, not only quick highlights

Plan:
Day 1-3: Helsinki
Day 4: Porvoo
Day 5-7: Turku and nearby coastal atmosphere
Day 8-9: Tampere
Day 10-13: Rovaniemi and Lapland
Day 14: Return south
Day 15: Departure

Route logic:
Longer Finland journey with regional variety, city culture, and northern nature.
""".strip()


def is_travel_request(user_input: str) -> bool:
    text = user_input.lower()
    keywords = [
        "trip", "travel", "itinerary", "route", "plan", "vacation", "holiday",
        "helsinki", "finland", "nordic", "lapland", "rovaniemi", "days"
    ]
    return any(word in text for word in keywords)


def choose_route_template(user_input: str) -> str:
    text = user_input.lower()

    # 非旅行内容，不用模板
    if not is_travel_request(text):
        return ""

    # 3天赫尔辛基
    if (
        "3-day" in text or "3 day" in text or "3 days" in text or "3days" in text
        or ("helsinki" in text and ("short" in text or "city break" in text or "3" in text))
    ):
        return ROUTE_3_DAY_HELSINKI

    # 7天芬兰 / 拉普兰
    if (
        "7-day" in text or "7 day" in text or "7 days" in text or "7days" in text
        or "lapland" in text or "rovaniemi" in text
    ):
        return ROUTE_7_DAY_FINLAND

    # 15天 / 家庭 / 从中国出发 / 更长行程
    if (
        "15-day" in text or "15 day" in text or "15 days" in text or "15days" in text
        or "family" in text or "from china" in text or "late april" in text
    ):
        return ROUTE_15_DAY_NORDIC

    # 一般旅行问题但未明确天数时，不强行塞 15 天模板
    return ""


def build_user_message(user_input: str, route_template: str) -> str:
    if not route_template:
        return user_input

    return f"""
User request:
{user_input}

Reference route template:
{route_template}

Task:
Answer as Nordic Travel AI.
If the user asked for an itinerary, provide a complete itinerary.
Do not stop at Day 1 or Day 2.
Keep the answer practical and concise.
""".strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "success": True,
        "message": "Flask backend is working."
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    非流式：保留给简单测试或备用
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "answer": "Error: request body is empty."}), 400

        user_input = str(data.get("input", "")).strip()
        if not user_input:
            return jsonify({"success": False, "answer": "Error: input is empty."}), 400

        route_template = choose_route_template(user_input)
        user_message = build_user_message(user_input, route_template)

        payload = {
            "model": MODEL_NAME,
            "stream": False,
            "think": False,
            "keep_alive": "15m",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "options": {
                "temperature": 0.4,
                "num_predict": 420
            }
        }

        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
        if response.status_code != 200:
            return jsonify({
                "success": False,
                "answer": f"Ollama HTTP error {response.status_code}: {response.text}"
            }), 500

        result = response.json()
        answer = result.get("message", {}).get("content", "").strip()

        if not answer:
            return jsonify({
                "success": False,
                "answer": f"Model returned empty response. Full result: {result}"
            }), 500

        return jsonify({"success": True, "answer": answer})

    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "answer": "Cannot connect to Ollama. Make sure Ollama is running."}), 500
    except requests.exceptions.Timeout:
        return jsonify({"success": False, "answer": "Ollama request timed out."}), 500
    except Exception as e:
        return jsonify({"success": False, "answer": f"Unexpected server error: {str(e)}"}), 500


@app.route("/generate_stream", methods=["POST"])
def generate_stream():
    """
    流式输出版本
    """
    data = request.get_json(silent=True)
    if not data:
        return Response("Error: request body is empty.", mimetype="text/plain")

    user_input = str(data.get("input", "")).strip()
    if not user_input:
        return Response("Error: input is empty.", mimetype="text/plain")

    route_template = choose_route_template(user_input)
    user_message = build_user_message(user_input, route_template)

    payload = {
        "model": MODEL_NAME,
        "stream": True,
        "think": False,
        "keep_alive": "15m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "options": {
            "temperature": 0.4,
            "num_predict": 420
        }
    }

    def generate_chunks():
        try:
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

        except requests.exceptions.ConnectionError:
            yield "\n\n[Error: Cannot connect to Ollama. Make sure Ollama is running.]"
        except requests.exceptions.Timeout:
            yield "\n\n[Error: Ollama request timed out.]"
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

    return Response(stream_with_context(generate_chunks()), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)
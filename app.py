from flask import Flask, request, jsonify
import openai, os, json, numpy as np, faiss

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAQs
with open("ReservAway_FAQs_AI.json") as f:
    faqs = json.load(f)

questions = [faq['question']['en'] for faq in faqs]
answers = [faq['answer']['en'] for faq in faqs]

# Embed function
def embed(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response['data'][0]['embedding'], dtype='float32')

embeddings = np.array([embed(q) for q in questions], dtype='float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

app = Flask(__name__)

@app.route('/raffy', methods=['POST'])
def raffy():
    data = request.json
    user_message = data.get("content", "")

    user_emb = embed(user_message).reshape(1, -1)
    D, I = index.search(user_emb, 1)

    if D[0][0] > 0.7:
        return jsonify({"content": "I’m not sure I have that information right now. [[ESCALATE]]"})

    best_answer = answers[I[0][0]]
    prompt = f"You are Raffy, an AI assistant for ReservAway. Use this FAQ answer to respond naturally:\nAnswer: {best_answer}\nUser: {user_message}\nRaffy:"
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=200, temperature=0)
    reply = response.choices[0].text.strip()

    return jsonify({"content": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

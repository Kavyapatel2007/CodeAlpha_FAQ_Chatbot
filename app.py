from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

faq = pd.read_csv("faq.csv")

questions = faq["Question"].tolist()
answers = faq["Answer"].tolist()

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_response(user_query):
    query_vector = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vector, question_vectors)
    best_match = similarity.argmax()

    if similarity[0][best_match] > 0.2:
        return answers[best_match]
    else:
        return "Sorry, I couldn't find an answer."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = get_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
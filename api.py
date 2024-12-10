from flask import Flask, request, jsonify
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import json

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = "trained_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the dataset
with open("qa_model_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)["data"]

# Helper function to calculate similarity
def calculate_similarity(question1, question2):
    tokens1 = set(question1.lower().split())
    tokens2 = set(question2.lower().split())
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

# Find the top 3 closest matching questions from the dataset
def find_top_matches(question, top_k=3):
    matches = []
    for entry in dataset:
        score = calculate_similarity(question, entry["question"])
        matches.append((entry, score))
    matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches[:top_k]]

@app.route("/answer", methods=["POST"])
def answer_question():
    # Get input question
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "A question is required"}), 400

    # Find the top matches
    top_matches = find_top_matches(question)
    if not top_matches:
        return jsonify({"error": "No matches found in the dataset"}), 404

    # Combine contexts for more generalized input
    combined_context = " ".join([match["context"] for match in top_matches])

    # Use the model to generate a dynamic response
    inputs = tokenizer(combined_context, question, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)

    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax() + 1
    generated_response = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])

    # Use the best match for ID and question
    best_match = top_matches[0]
    closest_id = best_match["id"]
    closest_question = best_match["question"]

    # Return the response
    return jsonify({
        "id": closest_id,
        "closest_question": closest_question,
        "generated_response": generated_response
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)


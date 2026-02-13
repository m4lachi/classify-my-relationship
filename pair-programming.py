from flask import Flask, render_template, request, jsonify  # pyright: ignore[reportMissingImports]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re  

app = Flask(__name__)

# =============================================================
# PAIR PROGRAMMING OVERVIEW (Backend)
# -------------------------------------------------------------
# Suggested flow:
#   Round 1:
#     - Driver: reads code out loud and scrolls
#     - Navigator: explains what each section does in plain English
#
#   Round 2 (TODO tasks below):
#     - Trade roles!
#     - Work through "PAIR PROGRAMMING TASK" comments together.
#
# You DO NOT have to finish all tasks. Pick 1–2 that feel doable.
# =============================================================


# Minimal defaults so the model always has at least one red + one green
DEFAULT_TRAINING = [
    {"text": "They respect your boundaries when you say no", "label": "green"},
    {"text": "They ignore your boundaries after you say no", "label": "red"},
]

# =============================================================
# PAIR PROGRAMMING TASK #1 (BACKEND, EASY)
# -------------------------------------------------------------
# With your partner:
#   - Add 1–3 more DEFAULT_TRAINING examples (mix of red + green).
#   - Think about "extreme" cases (like SUPER red or SUPER green).
#   - How might that affect the model's behavior?
#
# Example idea:
#   {"text": "They support your career even if it means long distance", "label": "green"}
# =============================================================


STOPWORDS = {
    "i", "me", "my", "mine", "you", "your", "yours",
    "they", "them", "their", "theirs",
    "we", "us", "our", "ours",
    "a", "an", "the", "and", "or", "but",
    "to", "for", "of", "in", "on", "at", "with", "from",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "doing",
    "this", "that", "these", "those",
    "he", "him", "his", "she", "her", "hers", "it", "its"
}

# =============================================================
# PAIR PROGRAMMING TASK #2 (BACKEND, MEDIUM)
# -------------------------------------------------------------
# STOPWORDS are words we ignore when computing contributions.
#
# With your partner:
#   - Look at this list and discuss:
#       * Are there any words you’d REMOVE from stopwords because
#         they might actually matter in relationship traits?
#         (Example: "we", "us" might be meaningful.)
#       * Are there any you’d ADD? (Ex: filler words you see a lot.)
#   - Try editing this set and see how it changes the chart.
# =============================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train_and_predict", methods=["POST"])
def train_and_predict():
    """
    Expects JSON:
    {
      "answers": [
        {"text": "Trait sentence 1", "label": "red"},
        {"text": "Trait sentence 2", "label": "green"},
        ...
      ],
      "user_trait": "Some new trait to classify"
    }
    """
    data = request.get_json(silent=True) or {}
    answers = data.get("answers", [])
    user_trait = data.get("user_trait", "").strip()

    # If somehow no answers came through, fall back to defaults
    if not answers:
        answers = DEFAULT_TRAINING.copy()

    if not user_trait:
        return jsonify({"error": "You must enter a trait"}), 400

    # ---------------------------------------------------------
    # PHASE 1: BUILD TRAINING DATA
    # ---------------------------------------------------------
    user_texts = [a["text"] for a in answers]
    user_labels = [a["label"] for a in answers]

    # Ensure both classes exist to avoid single-class training errors
    labels_set = set(user_labels)
    if "red" not in labels_set:
        user_texts.append(DEFAULT_TRAINING[1]["text"])
        user_labels.append("red")
    if "green" not in labels_set:
        user_texts.append(DEFAULT_TRAINING[0]["text"])
        user_labels.append("green")

    df = pd.DataFrame({"text": user_texts, "label": user_labels})

    # ---------------------------------------------------------
    # PHASE 2: VECTORIZER + MODEL
    # ---------------------------------------------------------
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Predict for the user's custom trait
    X_new = vectorizer.transform([user_trait])
    pred = model.predict(X_new)[0]

    # Prediction probabilities (for confidence display)
    probs = model.predict_proba(X_new)[0]
    class_indices = list(model.classes_)  # e.g. ['green', 'red']
    prob_dict = {label: float(prob) for label, prob in zip(class_indices, probs)}

    # ---------------------------------------------------------
    # PHASE 3: WORD-LEVEL CONTRIBUTIONS (EXPLAINABILITY)
    # ---------------------------------------------------------
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_          # shape: (n_classes or 1, n_features)
    classes = model.classes_     # e.g. array(['green', 'red'], dtype='<U5')

    vocab_index = {word: idx for idx, word in enumerate(feature_names)}

    raw_tokens = re.findall(r"[a-zA-Z']+", user_trait.lower())

    tokens = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        if len(tok) <= 2:
            continue
        tokens.append(tok)

    contributions: dict[str, dict[str, float]] = {}

    # Binary logistic regression case:
    # scikit-learn returns coefs with shape (1, n_features) for 2 classes.
    if coefs.shape[0] == 1 and len(classes) == 2:
        # row 0 is coef for log-odds of classes[1] vs classes[0]
        coef_row = coefs[0]
        pos_label = str(classes[1])  # class with positive direction of coef
        neg_label = str(classes[0])

        for tok in tokens:
            if tok in vocab_index:
                j = vocab_index[tok]
                w = float(coef_row[j])
                # Positive weight pushes toward pos_label, negative toward neg_label
                contributions[tok] = {
                    neg_label: -w,
                    pos_label: w,
                }
    else:
        # Multiclass/fallback: one row of coefs per class
        for tok in tokens:
            if tok in vocab_index:
                j = vocab_index[tok]
                contributions[tok] = {
                    str(label): float(coefs[i, j])
                    for i, label in enumerate(classes)
                }

    # =========================================================
    # PAIR PROGRAMMING TASK #3 (BACKEND, MEDIUM/HARD)
    # ---------------------------------------------------------
    # With your partner:
    #   - Right now, we only return prediction, probabilities,
    #     and contributions.
    #   - Brainstorm ONE extra thing you'd like to show in the UI:
    #       * Example: top 3 training sentences that influenced
    #         this decision (hint: nearest neighbors).
    #       * Or: a warning if the model is low-confidence.
    #   - Decide what extra field you’d add to this JSON response.
    #   - (Stretch) Try to add a new key here and log it in JS.
    # =========================================================

    return jsonify({
        "prediction": pred,            # "red" or "green"
        "probabilities": prob_dict,    # {"red": 0.73, "green": 0.27}
        "contributions": contributions
        # e.g. {"listen": {"red": 1.2, "green": -1.2}, ...}
    })


if __name__ == "__main__":
    app.run(debug=True)

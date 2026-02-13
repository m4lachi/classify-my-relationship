# 💘 Classify My Relationship

*A Valentine’s Machine Learning Workshop Project*

Train your own AI Cupid to spot 🚩 red flags and 💚 green flags, based on **your** definition of healthy relationships.

This project demonstrates:

* Supervised Machine Learning
* Text classification (Logistic Regression)
* Bias in AI systems
* Model interpretability using word contribution charts
* Real-time training with user-labeled data

---

## 🧠 What This Project Teaches

By playing the red/green flag game, you:

* Create your own labeled dataset
* Train a supervised ML classifier
* Predict new relationship traits
* Visualize which words influenced the decision
* See how bias forms in real time

> ML is just pattern recognition — and you’re teaching it your patterns.

---

## ⚙️ Tech Stack

* **Python**
* **Flask** (backend)
* **Scikit-learn** (Logistic Regression + CountVectorizer)
* **Pandas**
* **Chart.js** (frontend contribution chart)
* HTML + CSS (Valentine pastel theme 💗)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---

### 2️⃣ (Optional) Create a virtual environment

#### Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install flask scikit-learn pandas
```

---

### 4️⃣ Run the app

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 🎮 How It Works

1. You label 25 relationship traits as 🚩 or 💚
2. The model trains on your answers
3. You enter your own trait
4. The model predicts red/green
5. A contribution chart shows which words influenced the decision

---

## ⚖️ Bias in AI

This project intentionally demonstrates:

* Models reflect the data they are trained on
* Small labeling differences change predictions
* Word-level bias impacts output
* AI is never neutral — it mirrors human input

This is a safe, playful way to explore real-world ML ethics.

---

## 💞 Pair Programming Challenges

Want to extend it?

Try one of these:

* Add more red/green traits
* Add a new category (e.g., 🟡 “Yellow Flag”)
* Add a reset button
* Add sound effects
* Add Google Form integration
* Improve the model (Naive Bayes, SVM, etc.)
* Add personality profiles

Switch Driver/Navigator roles halfway through 💻✨

---

## 🌸 Workshop Slides

Slides from the workshop:
👉 **[[Check out my slides here](https://www.canva.com/design/DAHA3aRRsZ0/u3peHbdGKCx98l9Wm5g52Q/edit?utm_content=DAHA3aRRsZ0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)]**

---

## 💗 Future Improvements

* Store training data in a database
* Use word embeddings instead of bag-of-words
* Add model comparison
* Host online (Render / Replit)
* Add user login and profiles

---

## 💌 Created For

Women in Computing Society (WiCS)
Valentine’s Machine Learning Workshop

Built with code and a little bit of curiosity  💘

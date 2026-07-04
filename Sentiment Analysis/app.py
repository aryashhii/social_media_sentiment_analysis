from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

HISTORY_FILE = "predictions.csv"

if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(columns=["Platform", "Comment", "Prediction"])
    df.to_csv(HISTORY_FILE, index=False)


@app.route("/")
def home():
    try:
        history = pd.read_csv(HISTORY_FILE)
    except Exception:
        history = pd.DataFrame(columns=["Platform", "Comment", "Prediction"])

    if history.empty:
        history = pd.DataFrame(columns=["Platform", "Comment", "Prediction"])

    total_comments = len(history)
    total_positive = len(history[history["Prediction"] == "Positive"])
    total_negative = len(history[history["Prediction"] == "Negative"])

    def positive_percentage(platform):
        data = history[history["Platform"] == platform]
        if len(data) == 0:
            return 0
        positive = len(data[data["Prediction"] == "Positive"])
        return round((positive / len(data)) * 100, 1)

    youtube_percent = positive_percentage("YouTube")
    instagram_percent = positive_percentage("Instagram")
    facebook_percent = positive_percentage("Facebook")

    return render_template(
        "index.html",
        prediction=None,
        sentence=None,
        history=history.to_dict("records"),
        total_comments=total_comments,
        total_positive=total_positive,
        total_negative=total_negative,
        youtube_percent=youtube_percent,
        instagram_percent=instagram_percent,
        facebook_percent=facebook_percent
    )


@app.route("/predict", methods=["POST"])
def predict():
    platform = request.form["platform"]
    sentence = request.form["sentence"]

    sentence_vector = vectorizer.transform([sentence])

    prediction = model.predict(sentence_vector)[0]

    try:
        history = pd.read_csv(HISTORY_FILE)
    except Exception:
        history = pd.DataFrame(columns=["Platform", "Comment", "Prediction"])

    if history.empty:
        history = pd.DataFrame(columns=["Platform", "Comment", "Prediction"])

    new_data = pd.DataFrame({
        "Platform": [platform],
        "Comment": [sentence],
        "Prediction": [prediction]
    })

    history = pd.concat([history, new_data], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

    total_comments = len(history)
    total_positive = len(history[history["Prediction"] == "Positive"])
    total_negative = len(history[history["Prediction"] == "Negative"])

    def positive_percentage(platform_name):
        data = history[history["Platform"] == platform_name]
        if len(data) == 0:
            return 0
        positive = len(data[data["Prediction"] == "Positive"])
        return round((positive / len(data)) * 100, 1)

    youtube_percent = positive_percentage("YouTube")
    instagram_percent = positive_percentage("Instagram")
    facebook_percent = positive_percentage("Facebook")

    return render_template(
        "index.html",
        prediction=prediction,
        sentence=sentence,
        history=history.to_dict("records"),
        total_comments=total_comments,
        total_positive=total_positive,
        total_negative=total_negative,
        youtube_percent=youtube_percent,
        instagram_percent=instagram_percent,
        facebook_percent=facebook_percent
    )


if __name__ == "__main__":
    app.run(debug=True)

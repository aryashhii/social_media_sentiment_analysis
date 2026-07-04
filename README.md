An interactive Flask-based web application that performs sentiment analysis on social media comments (YouTube, Instagram, Facebook) using machine learning.
The dashboard provides real-time predictions, analytics, and visualizations to help track public opinion across platforms.

Features :-
> Sentiment Prediction:
Classifies comments as Positive or Negative using a trained ML model.

> Interactive Dashboard:
Input comments and select platform.
View prediction results instantly.

> Analytics:
Total comments processed.
Positive vs Negative counts.
Platform-wise positive sentiment percentages.

> History Tracking:
Stores all predictions in a CSV file (predictions.csv).
Displays prediction history in a table.

> Visualizations:
Bar chart showing sentiment distribution across platforms.
Progress bars for platform-specific analytics.

🛠️ Tech Stack
Backend: Flask (Python)
Frontend: HTML, CSS, Jinja2, Chart.js
Machine Learning: scikit-learn (Logistic Regression, Naive Bayes, SVM)
Data Handling: Pandas, Joblib
Storage: CSV file for prediction history

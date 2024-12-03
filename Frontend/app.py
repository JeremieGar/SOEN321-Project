from flask import Flask, render_template, request, send_from_directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'privacyiqmain')))
from privacyiqmain.preprocess_policy import preprocess_policy_file

app = Flask(__name__, static_folder='static')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text input from the form
        policy_text = request.form['policy_text']

        if policy_text:

            processed_text = preprocess_policy_file(policy_text)  # This will be adjusted based on the process

            total_score = 0
            count = 0
            risk_scores = {"No risk": 3, "Low risk": 2, "Medium risk": 1, "High risk": 0}

            for entry in processed_text:
                label = entry.get("label")
                if label in risk_scores:
                    total_score += risk_scores[label]
                    count += 1

            average_score = round(float(total_score / count) if count > 0 else 0)

            return render_template('results.html', processed_text=processed_text, average_score=average_score)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
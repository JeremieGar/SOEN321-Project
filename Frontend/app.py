from flask import Flask, render_template, request, send_from_directory
import sys
import os
# Add the directory that contains privacyiqmain to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'privacyiqmain')))
from privacyiqmain.preprocess_policy import preprocess_policy_file

app = Flask(__name__, static_folder='static')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text input from the form
        policy_text = request.form['policy_text']

        if policy_text:
            # Process the text (you can pass it to your preprocessing function here)
            # For now, we simulate the processing by just passing it to the results page
            processed_text = preprocess_policy_file(policy_text)  # This will be adjusted based on the process

            # Render the results.html page and pass the processed data
            return render_template('results.html', processed_text=processed_text)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
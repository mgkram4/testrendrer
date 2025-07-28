import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Dummy data for demonstration
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

# Train a simple Logistic Regression model
X = df[['feature1', 'feature2']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    feature1 = data['feature1']
    feature2 = data['feature2']
    
    # Load the model
    with open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
        
    prediction = model.predict([[feature1, feature2]])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True) 
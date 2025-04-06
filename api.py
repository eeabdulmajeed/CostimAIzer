from flask import Flask, request, jsonify
from backend.cost_estimator import CostEstimator

app = Flask(__name__)

@app.route('/estimate', methods=['POST'])
def estimate():
    data = request.json
    task_description = data.get('task_description')
    estimator = CostEstimator()
    result = estimator.analyze_and_estimate(task_description)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
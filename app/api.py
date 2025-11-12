"""
App REST API endpoints (could be separate from Flask main)
"""
from flask import Blueprint, request, jsonify
api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST'])
def predict():
    # Placeholder: forward to main.py logic
    return jsonify({'status': 'ok', 'msg': 'API ready'})

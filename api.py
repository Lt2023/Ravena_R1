import http.client
import json
from flask import Flask, request, jsonify
from model import LanguageModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

model = LanguageModel()

def verify_ca(ca_value):
    try:
        conn = http.client.HTTPSConnection("ai.coludai.cn")
        
        payload = json.dumps({"ca": ca_value})
        
        headers = {'Content-Type': 'application/json'}
        
        conn.request("POST", "/api/ca/verify", payload, headers)
        
        res = conn.getresponse()
        data = res.read()
        
        response_data = json.loads(data.decode("utf-8"))
        
        if response_data.get("success"):
            return True
        else:
            return False
    except Exception as e:
        print(f"验证 CA 失败: {e}")
        return False

@app.route('/ask', methods=['POST'])
def ask():
    ca_value = request.headers.get('ca')

    if not ca_value:
        return jsonify({"error": "未提供 CA 请求头"}), 400

    if not verify_ca(ca_value):
        return jsonify({"error": "CA 验证失败，拒绝服务"}), 403
    
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "没有提供问题"}), 400

    response = model.generate_answer(question)

    return jsonify({"answer": response})

@app.route('/')
def home():
    return "Ravena_4由😘刘时安&ColudAI开发"

if __name__ == "__main__":
    print("服务已启动，正在监听端口 5000...")
    app.run(debug=True, host="0.0.0.0", port=5000)

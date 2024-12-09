import requests
import json

# 测试 API 地址
BASE_URL = "http://localhost:5000"

# 模拟的 CA 验证值，可以替换为有效的 CA 或无效的 CA 进行测试
CA_VALID = ""   # 替换为有效的 CA 值
CA_INVALID = "invalid-ca-value"    # 用一个无效的 CA 值测试失败的情况

# 简化的测试函数：验证 CA 成功的情况
def test_ca_success():
    url = f"{BASE_URL}/ask"
    headers = {
        'Content-Type': 'application/json',
        'ca': CA_VALID
    }
    # 在这里填写问题
    payload = {
        "question": "你是谁？"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    # 只打印 API 返回的回答
    if response.status_code == 200:
        print(response.json().get("answer"))
    else:
        print(f"错误: {response.json().get('error')}")

# 简化的测试函数：验证 CA 失败的情况
def test_ca_failure():
    url = f"{BASE_URL}/ask"
    headers = {
        'Content-Type': 'application/json',
        'ca': CA_VALID
    }
    # 在这里填写问题
    payload = {
        "question": "你好"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    # 只打印 API 返回的回答或错误信息
    if response.status_code == 200:
        print(response.json().get("answer"))
    else:
        print(f"错误: {response.json().get('error')}")

# 简化的测试函数：没有 CA 请求头的情况
def test_no_ca_header():
    url = f"{BASE_URL}/ask"
    headers = {
        'Content-Type': 'application/json'
    }
    # 在这里填写问题
    payload = {
        "question": "你好"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    # 只打印 API 返回的回答或错误信息
    if response.status_code == 200:
        print(response.json().get("answer"))
    else:
        print(f"错误: {response.json().get('error')}")

# 简化的测试函数：没有问题字段的情况
def test_no_question():
    url = f"{BASE_URL}/ask"
    headers = {
        'Content-Type': 'application/json',
        'ca': CA_VALID
    }
    payload = {}  # 没有 question 字段
    response = requests.post(url, json=payload, headers=headers)
    
    # 只打印 API 返回的回答或错误信息
    if response.status_code == 200:
        print(response.json().get("answer"))
    else:
        print(f"错误: {response.json().get('error')}")

# 主函数，执行所有测试
if __name__ == "__main__":
    test_ca_success()
    test_ca_failure()
    test_no_ca_header()
    test_no_question()

import requests

def query_wikipedia(word):
    # 维基百科API的URL
    url = "https://en.wikipedia.org/w/api.php"

    # 设置请求参数
    params = {
        "action": "query",
        "format": "json",
        "titles": word,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }

    # 发送请求
    response = requests.get(url, params=params)
    data = response.json()

    # 解析返回的数据
    page = next(iter(data["query"]["pages"].values()))
    extract = page.get("extract", "No description available.")

    return extract

# 示例用法
word = "Python_(programming_language)"
description = query_wikipedia(word)
print(description)


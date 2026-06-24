import wikipediaapi

def get_summary(word, lang='en', summary_length=100):
    # 创建Wikipedia对象
    # 创建一个Wikipedia对象并指定用户代理
    wiki_wiki = wikipediaapi.Wikipedia('en', user_agent="my")

    # 使用该对象获取页面
    page = wiki_wiki.page("Python_(programming_language)")

    # 检查页面是否存在
    if not page.exists():
        return "Page not found for " + word

    # 获取简短摘要
    summary = page.summary[:summary_length]
    # 保证不在中间的句子结束
    last_period = summary.rfind(".")
    if last_period != -1:
        summary = summary[:last_period + 1]

    return summary

# 十个单词的列表
words = ["Python", "Computer", "Internet", "Art", "Music", "Physics", "Mathematics", "Literature", "Engineering", "Philosophy"]

# 获取每个单词的摘要
summaries = {word: get_summary(word) for word in words}

for word, summary in summaries.items():
    print(f"{word}: {summary}\n")

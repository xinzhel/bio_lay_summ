import json

def read_data(file_path):
    with open(file_path, 'r') as f:
        lst = []
        for line in f.readlines():
            text = json.loads(line)
            lst.append(text)
    print("Read data from : ", file_path)
    print("The number of data: ", len(lst))

    # articles and summaries
    articles = [example['article'] for example in lst]
    summaries = [example['lay_summary'] for example in lst]

    return articles, summaries

def read_instances(file_path):
    with open(file_path, 'r') as f:
        lst = []
        for line in f.readlines():
            text = json.loads(line)
            lst.append(text)
    print("Read data from : ", file_path)
    print("The number of data: ", len(lst))
    return lst


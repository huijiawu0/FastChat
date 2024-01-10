import json
import os
import uuid
from collections import defaultdict, OrderedDict
from pprint import pprint

import pandas as pd
from io import StringIO

from flask import Flask, request, jsonify
import subprocess
import random
import string
import time
import datetime
import pytz

from fastchat.llm_judge.gen_model_answer import run_eval
from fastchat.serve.flask.utils import calculate_model_scores, read_jsonl_files
from fastchat.utils import str_to_torch_dtype
from flask_utils import get_free_gpus, append_dict_to_jsonl, get_end_time, get_start_time
from fastchat.llm_judge.report.assist1 import generate_report, get_system_prompt, get_cache

app_dir = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(app_dir, 'resources', 'data_config.json')
with open(DATA_PATH) as file:
    DATA_JSON = json.load(file)
DATA_DICT = {dataset["data_id"]: dataset for dataset in DATA_JSON[0]["datasets"]}

MODEL_PATH = os.path.join(app_dir, 'resources', 'model_config.json')
with open(MODEL_PATH) as file:
    MODEL_JSON = json.load(file)
MODEL_DICT = {model["model_id"]: model for model in MODEL_JSON["models"]}


def generate_random_model_id():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


app = Flask(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@app.route('/get_modelpage_list', methods=['POST'])
def get_modelpage_list():
    request_id = random_uuid()
    result = MODEL_JSON.copy()
    result.update({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_modelpage_detail', methods=['POST'])
def get_modelpage_detail():
    request_id = random_uuid()
    data = request.json
    if not all(key in data for key in ['model_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    
    MODEL_ID = data.get('model_id')
    overall_report = calculate_model_scores(["moral_bench_test1"])
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = get_cache()
    ability_scores = overall_report[MODEL_ID]["score_per_category"]
    ability_scores_array = []
    for model, scores in ability_scores.items():
        model_scores = {"model": model}
        model_scores.update(scores)
        ability_scores_array.append(model_scores)
    result = {
        "request_id": request_id,
        "model_id": MODEL_ID,
        "score": overall_report[MODEL_ID]["score_total"],
        "ability_scores": ability_scores_array,
        "model_description": MODEL_DICT.get(MODEL_ID, {}),
        "report": report
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_datapage_list', methods=['POST'])
def get_datapage_list():
    request_id = random_uuid()
    result = DATA_JSON.copy()
    result.append({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_datapage_detail', methods=['POST'])
def get_datapage_detail():
    request_id = random_uuid()
    data = request.json
    if not all(key in data for key in ['data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    DATA_ID = data.get('data_id')
    # DATA_ID = "moral_bench_test1"
    overall_report = calculate_model_scores([DATA_ID])
    result = {
        "request_id": request_id,
        "data_id": DATA_ID,
        "data_description": DATA_DICT.get(DATA_ID, {}),
        "score": {model: item["score_total"] for model, item in overall_report.items()},
        "model_ids": list(overall_report.keys()),
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_leaderboard_detail', methods=['POST'])
def get_leaderboard_detail():
    request_id = random_uuid()
    result = {
        "request_id": request_id,
        "header": [
            "模型",
            "发布日期",
            "类型",
            "参数量",
            "综合",
            "政治伦理",
            "经济伦理",
            "社会伦理",
            "文化伦理",
            "科技伦理",
            "环境伦理",
            "医疗健康伦理",
            "教育伦理",
            "职业道德",
            "艺术与文化伦理",
            "网络与信息伦理",
            "国际关系与全球伦理",
            "心理伦理",
            "生物伦理",
            "运动伦理"
        ],
        "data": [
            {
                "模型": "ChatGLM2",
                "发布日期": "2023-01-01",
                "类型": "大语言模型",
                "参数量": 175000000,
                "综合": 85.25,
                "政治伦理": 92.00,
                "经济伦理": 87.75,
                "社会伦理": 88.50,
                "文化伦理": 84.25,
                "科技伦理": 89.00,
                "环境伦理": 86.50,
                "医疗健康伦理": 90.00,
                "教育伦理": 85.75,
                "职业道德": 88.25,
                "艺术与文化伦理": 82.75,
                "网络与信息伦理": 87.50,
                "国际关系与全球伦理": 89.25,
                "心理伦理": 91.00,
                "生物伦理": 88.75,
                "运动伦理": 84.00
            },
            {
                "模型": "示例模型2",
                "发布日期": "2023-02-15",
                "类型": "示例类型",
                "参数量": 1000000,
                "综合": 78.50,
                "政治伦理": 80.00,
                "经济伦理": 75.25,
                "社会伦理": 82.75,
                "文化伦理": 79.25,
                "科技伦理": 77.00,
                "环境伦理": 80.50,
                "医疗健康伦理": 85.00,
                "教育伦理": 76.25,
                "职业道德": 81.00,
                "艺术与文化伦理": 77.75,
                "网络与信息伦理": 79.00,
                "国际关系与全球伦理": 83.25,
                "心理伦理": 80.00,
                "生物伦理": 76.75,
                "运动伦理": 78.50
            }
        ]
    }
    return json.dumps(result, ensure_ascii=False)


def calculate_score(result_dict):
    score_result = {}
    for model, model_result in result_dict.items():
        category_status = defaultdict(list)
        for answer in model_result:
            category = answer["category"].split('|||')[0]
            pred = answer["choices"][0]["turns"][0].split('')[0]
            pred_counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
            refer_counts = {option: answer["reference_answer"].count(option) for option in ['A', 'B', 'C', 'D']}
            status = all(pred_counts[option] == refer_counts[option] for option in ['A', 'B', 'C', 'D'])
            category_status[category].append(status)
        
        category_score = {k: (sum(v) / len(v), sum(v), len(v)) for k, v in category_status.items()}
        total_correct = sum(v[1] for v in category_score.values())
        total_questions = sum(v[2] for v in category_score.values())
        score_result[model] = (
        total_correct, total_questions, total_correct / total_questions if total_questions else 0)
    
    return score_result


def get_total_scores(model_scores):
    total_scores = {}
    for model, scores in model_scores.items():
        total_scores[model] = sum(scores.values())
    return total_scores


@app.route('/get_report', methods=['POST'])
def get_report():
    data = request.json
    data_ids = data.get('data_ids')
    model_ids = data.get('model_ids')
    print(data_ids, model_ids)

    if not data_ids or not model_ids:
        return jsonify({"error": "Missing required fields in the request"}), 400
    
    base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    all_model_scores = defaultdict(lambda: defaultdict(int))
    for data_id in data_ids:
        directory_path = os.path.join(base_path, "llm_judge", "data", data_id, "model_answer")
        result_dict = read_jsonl_files(directory_path)
        model_scores = calculate_model_scores(result_dict)
        print(list(result_dict.keys()), model_scores)
        for model_id in model_ids:
            all_model_scores[model_id][data_id] = model_scores[model_id][data_id]

    total_scores = get_total_scores(all_model_scores)
    header = ['Model ID', 'Total Score'] + data_ids
    score_table = [header]
    for model_id in model_ids:
        row = [model_id, total_scores[model_id]] + [all_model_scores[model_id][data_id] for data_id in data_ids]
        score_table.append(row)

    return jsonify({"score_table": score_table})


@app.route('/run_evaluate', methods=['POST'])
def run_evaluate():
    data = request.json
    if not all(key in data for key in ['model_names', 'model_ids', 'data_ids']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    model_names = data.get('model_names')
    model_ids = data.get('model_ids')
    data_ids = data.get('data_ids')
    revision = data.get('revision', None)
    question_begin = data.get('question_begin', None)
    question_end = data.get('question_end', None)
    max_new_token = data.get('max_new_token', 1024)
    num_choices = data.get('num_choices', 1)
    num_gpus_per_model = data.get('num_gpus_per_model', 1)
    num_gpus_total = data.get('num_gpus_total', 1)
    max_gpu_memory = data.get('max_gpu_memory', 16)
    dtype = str_to_torch_dtype(data.get('dtype', None))
    cache_dir = data.get('cache_dir', "/root/autodl-tmp/model")
    base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    print("model_names:", model_names, "model_ids:", model_ids, "data_ids:", data_ids, "cache_dir:", cache_dir)
    try:
        start_time = get_start_time()
        outputs = []
        for data_id in data_ids:
            question_file = os.path.join(base_path, "llm_judge", "data", str(data_id), "question.jsonl")
            for model_name, model_id in zip(model_names, model_ids):
                output_file = os.path.join(base_path, "llm_judge", "data", str(data_id), "model_answer",
                                           f"{model_id}.jsonl")
                run_eval(
                    model_path=model_name,
                    model_id=model_id,
                    question_file=question_file,
                    question_begin=question_begin,
                    question_end=question_end,
                    answer_file=output_file,
                    max_new_token=max_new_token,
                    num_choices=num_choices,
                    num_gpus_per_model=num_gpus_per_model,
                    num_gpus_total=num_gpus_total,
                    max_gpu_memory=max_gpu_memory,
                    dtype=dtype,
                    revision=revision,
                    cache_dir=cache_dir
                )
                outputs.append(
                    {"data_id": data_id, "model_id": model_id, "model_name": model_name, "output": output_file})
        end_time = get_end_time()
        result = {"outputs": outputs,
                  "model_names": model_names,
                  "model_ids": model_ids,
                  "data_ids": data_ids,
                  "time_start": start_time,
                  "time_end": end_time}
        # append_dict_to_jsonl(f"/home/workspace/FastChat/fastchat/llm_judge/data/{data_id}/app_output.jsonl",
        #                      {model_id: result})
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5004)

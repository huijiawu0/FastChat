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
    DATA_IDS = list(DATA_DICT.keys())
    DATA_IDS.extend(["moral_bench_test1", "moral_bench_test2"])
    print("model_id:", MODEL_ID, "data_ids:", DATA_IDS)
    overall_report = calculate_model_scores(DATA_IDS)
    print("overall_report:", overall_report)
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = get_cache()

    ability_scores = overall_report[MODEL_ID]["score_per_category"]
    ability_scores_array = []
    for ability, scores in ability_scores.items():
        ability_scores_array.append({"ability": ability, **scores})

    scores_per_data_id = overall_report[MODEL_ID]["scores_per_data_id"]
    data_id_scores = []
    for data_id, scores in scores_per_data_id.items():
        data_id_scores.append(
            {"data_id": data_id, "score": scores["correct"], "total": scores["total"], "accuracy": scores["accuracy"]})
    result = {
        "request_id": str(request_id),
        "model_id": MODEL_ID,
        "score": overall_report[MODEL_ID]["score_total"],
        "correct": overall_report[MODEL_ID]["total_correct"],
        "total": overall_report[MODEL_ID]["total_questions"],
        "ability_scores": ability_scores_array,
        "data_id_scores": data_id_scores,
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
            "合规性",
            "公平性",
            "知识产权",
            "隐私保护",
            "可信度"
        ],
        "data": [
            {
                "模型": "ChatGLM2",
                "发布日期": "2023-01-01",
                "类型": "大语言模型",
                "参数量": "6B",
                "综合": 85.25,
                "合规性": 92.00,
                "公平性": 87.75,
                "知识产权": 88.50,
                "隐私保护": 84.25,
                "可信度": 89.00
            },
            {
                "模型": "ChatGLM3",
                "发布日期": "2023-02-15",
                "类型": "大语言模型",
                "参数量": "7B",
                "综合": 78.50,
                "合规性": 80.00,
                "公平性": 75.25,
                "知识产权": 82.75,
                "隐私保护": 79.25,
                "可信度": 77.00
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
    request_id = random_uuid()
    data = request.json
    data_ids = data.get('data_ids')
    model_ids = data.get('model_ids')
    print(data_ids, model_ids)

    if not data_ids or not model_ids:
        return jsonify({"error": "Missing required fields in the request"}), 400
    
    report = calculate_model_scores(data_ids)
    
    header = ['Model ID', 'Total Score'] + data_ids + ["Evaluate Time", "Report"]
    leaderboard = [header]
    for model, model_data in report.items():
        row = [model]
        total_correct = model_data['total_correct']
        total_questions = model_data['total_questions']
        total_score = total_correct / total_questions if total_questions > 0 else 0
        row.append(total_score)
        for data_id in data_ids:
            score_per_data_id = model_data['scores_per_data_id'].get(data_id, {"correct": 0, "total": 0})
            category_score = score_per_data_id['correct'] / score_per_data_id['total'] \
                if score_per_data_id['total'] > 0 else 0
            row.append(category_score)
        report = get_cache()
        row.append(get_end_time())
        row.append(report)
        leaderboard.append(row)

    return json.dumps({"request_id":request_id, "leaderboard": leaderboard}, ensure_ascii=False)


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

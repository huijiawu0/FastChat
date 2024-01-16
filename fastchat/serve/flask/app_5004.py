import json
import os
import uuid
from collections import defaultdict, OrderedDict
from pprint import pprint

import pandas as pd
from io import StringIO

import torch
from flask import Flask, request, jsonify
import subprocess
import random
import string
import time
import datetime
import pytz

from fastchat.llm_judge.gen_model_answer import run_eval
from fastchat.serve.flask.utils import calculate_model_scores, read_jsonl_files, calculate_model_scores2
from fastchat.utils import str_to_torch_dtype
from flask_utils import get_free_gpus, append_dict_to_jsonl, get_end_time, get_start_time
from fastchat.llm_judge.report.assist1 import generate_report, get_system_prompt, get_cache

app_dir = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(app_dir, 'resources', 'data_config.json')
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    DATA_JSON = json.load(file)
DATA_DICT = {}
for DATA_CATEGORY in DATA_JSON:
    for DATA in DATA_CATEGORY['datasets']:
        DATA_DICT[DATA['data_id']] = DATA
DATA_IDS = [dataset["data_id"] for dataset in DATA_JSON[0]["datasets"]]
MODEL_PATH = os.path.join(app_dir, 'resources', 'model_config.json')
with open(MODEL_PATH) as file:
    MODEL_JSON = json.load(file)
MODEL_DICT = {model["name"].split('/')[-1]: model for model in MODEL_JSON["models"]}
MODEL_NAMES = [model['name'] for model in MODEL_JSON["models"]]
MODEL_IDS = [model['model_id'] for model in MODEL_JSON["models"]]
BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print("BASE_PATH:", BASE_PATH)
print("DATA_PATH:", DATA_PATH)
print("MODEL_PATH:", MODEL_PATH)
RENAME_DATA = {
    'political_ethics_dataset': '政治伦理',
    'economic_ethics_dataset': '经济伦理',
    'social_ethics_dataset': '社会伦理',
    'cultural_ethics_dataset': '文化伦理',
    'technology_ethics_dataset': '科技伦理',
    'environmental_ethics_dataset': '环境伦理',
    'medical_ethics_dataset': '医疗健康伦理',
    'education_ethics_dataset': '教育伦理',
    'professional_ethics_dataset': '职业道德伦理',
    'cyber_information_ethics_dataset': '网络伦理',
    'international_relations_ethics_dataset': '国际关系与全球伦理',
    'psychology_ethics_dataset': '心理伦理',
    'bioethics_dataset': '生物伦理学',
    'sports_ethics_dataset': '运动伦理学',
    'military_ethics_dataset': '军事伦理'
}


def generate_random_model_id():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


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


def get_report_by_names(request_id, data_ids, model_names):
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    categories = ['合规性', '公平性', '知识产权', '隐私保护', '可信度']
    header = ['Model ID', 'Total Score'] + categories + ["Evaluate Time", "Report"]
    leaderboard = [header]
    for model, model_data in report_per_model.items():
        if model not in model_names:
            print("model not in model_names:", model, model_names)
            continue
        else:
            row = [model]
            total_correct = model_data['total_correct']
            total_questions = model_data['total_questions']
            total_score = total_correct / total_questions if total_questions > 0 else 0
            row.append(total_score)
            for category in categories:
                score_per_category_id = model_data['score_per_category'].get(category, {"correct": 0, "total": 0})
                category_score = score_per_category_id['correct'] / score_per_category_id['total'] \
                    if score_per_category_id['total'] > 0 else 0
                row.append(category_score)
            # report = get_cache()
            report = ""
            row.append(get_end_time())
            row.append(report)
            leaderboard.append(row)
    return json.dumps({"request_id": request_id, "leaderboard": leaderboard}, ensure_ascii=False)


def get_report_all():
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    result = {}
    for model, model_data in report_per_model.items():
        total_correct = model_data['total_correct']
        total_questions = model_data['total_questions']
        total_score = total_correct / total_questions if total_questions > 0 else 0
        report = get_cache()
        model_data.update({"Total Score": total_score, "Report": report, "Evaluate Time": get_end_time()})
        result.update({model: model_data})
    return result


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


app = Flask(__name__)


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
    if not all(key in data for key in ['model_name']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    
    MODEL_NAME = data.get('model_name')
    DATA_IDS = list(DATA_DICT.keys())
    print("model_name:", MODEL_NAME, "data_ids:", DATA_IDS)
    # overall_report = calculate_model_scores(DATA_IDS)
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    print("report_per_model:", report_per_model)
    print("report_per_data:", report_per_data)
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = get_cache()
    try:
        MODEL_NAME = MODEL_NAME.split('/')[-1] if MODEL_NAME not in report_per_model else MODEL_NAME
    except AttributeError as e:
        print(e)
        return jsonify({"error": f"Model NAME '{MODEL_NAME}' not found in the report", "code": "ModelNotFound"}), 404
    if MODEL_NAME not in report_per_model:
        return jsonify({"error": f"Model NAME '{MODEL_NAME}' not found in the report", "code": "ModelNotFound"}), 404
    else:
        ability_scores = report_per_model[MODEL_NAME]["score_per_category"]
        ability_scores_array = []
        for ability, scores in ability_scores.items():
            ability_scores_array.append({"ability": ability, **scores})
  
        scores_per_data_id = report_per_model[MODEL_NAME]["scores_per_data_id"]
        data_id_scores = []
        for data_id, scores in scores_per_data_id.items():
            data_id_scores.append(
                {"data_id": data_id, "score": scores["correct"], "total": scores["total"],
                 "accuracy": scores["accuracy"]})
        result = {
            "request_id": str(request_id),
            "model_name": MODEL_NAME,
            "score": report_per_model[MODEL_NAME]["score_total"],
            "correct": report_per_model[MODEL_NAME]["total_correct"],
            "total": report_per_model[MODEL_NAME]["total_questions"],
            "ability_scores": ability_scores_array,
            "data_id_scores": data_id_scores,
            "model_description": MODEL_DICT.get(MODEL_NAME, {}),
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
    DATA_RENAME = RENAME_DATA.get(DATA_ID, None)
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    
    result = {
        "request_id": request_id,
        "data_id": DATA_ID,
        "data_description": DATA_DICT.get(DATA_ID, {}),
        "score": report_per_data.get(DATA_RENAME, 0),
        "model_ids": list(report_per_model.keys()),
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_leaderboard_detail', methods=['POST'])
def get_leaderboard_detail():
    CATEGORY = ["合规性", "公平性", "知识产权", "隐私保护", "可信度"]
    filter_params = request.json
    categories = filter_params.get('categories', None)
    if categories is None:
        categories = CATEGORY.copy()
    model_sizes = filter_params.get('model_sizes', None)
    datasets = filter_params.get('datasets', None)
    print("categories:", categories, "model_sizes:", model_sizes, "datasets:", datasets)
    filtered_cates = CATEGORY.copy()
    if categories is not None:
        filtered_cates = [cate for cate in CATEGORY if cate in categories]
    filtered_models = [model["name"].split('/')[-1] for model in MODEL_JSON["models"]]
    if model_sizes is not None:
        filtered_models = [model for model in filtered_models if
                           any(size.lower() in model.lower() for size in model_sizes)]
    filtered_data = ["moral_bench_test5"]
    print("filtered_cates:", filtered_cates, "filtered_models:", filtered_models, "filtered_data:", filtered_data)
    
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    aggregated_scores = {}
    for model_name in filtered_models:
        if model_name not in report_per_model:
            print("model_name not in report_per_model:", model_name)
            continue
        else:
            model_data = report_per_model[model_name]
            aggregated_scores[model_name] = {category: 0 for category in categories}
            aggregated_scores[model_name]['count'] = 0
    
            for category in categories:
                category_score = model_data['score_per_category'].get(category, {})
                aggregated_scores[model_name][category] = category_score.get('accuracy', 0)

            aggregated_scores[model_name]['count'] = model_data['total_questions']

    print("aggregated_scores:", aggregated_scores)

    final_data = []
    for model_name, scores in aggregated_scores.items():
        if model_name in filtered_models:
            avg_scores = {cat: scores[cat] for cat in categories}
            final_data.append({
                "模型": model_name,
                "综合": sum(avg_scores.values()) / len(categories),
                **avg_scores
            })
    print("final_data:", final_data)
    result = {
        "request_id": str(uuid.uuid4()),
        "header": [
                      "模型", "综合"
                  ] + categories,
        "data": final_data
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_report', methods=['POST'])
def get_report():
    def get_evaluation_results(request_id):
        log_folder = os.path.join(BASE_PATH, "llm_judge", "log")
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, "eval_log.jsonl")
        with open(log_path, 'r') as f:
            for line in f:
                js0 = json.loads(line)
                if request_id in js0:
                    return js0[request_id]
        return None
    
    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return jsonify({"error": "Missing request_id in the request"}), 400
    
    evaluation_results = get_evaluation_results(request_id)
    print("evaluation_results:", evaluation_results)
    if evaluation_results is not None:
        data_ids = evaluation_results["data_ids"]
        model_names = [model_name.split('/')[-1] for model_name in evaluation_results["model_names"]]
        print(__name__, "data_ids:", data_ids, "model_names:", model_names)
        return get_report_by_names(request_id, data_ids, model_names)
    else:
        return jsonify({"error": f"No evaluation results found by request_id {request_id}"}), 400


def is_non_empty_file(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0


@app.route('/run_evaluate', methods=['POST'])
def run_evaluate():
    global ray
    request_id = random_uuid()
    data = request.json
    model_names = data.get('model_names', None)
    model_ids = data.get('model_ids', None)
    data_ids = data.get('data_ids', None)
    if model_names is None or model_ids is None:
        model_names = MODEL_NAMES
        model_ids = MODEL_IDS
    if data_ids is None:
        data_ids = DATA_IDS
        print("using default settings", model_names, model_ids, data_ids)
    if len(model_names) != len(model_ids):
        print(model_names, model_ids)
        return jsonify({"error": "model_names and model_ids should have the same length"}), 400
    
    revision = data.get('revision', None)
    question_begin = data.get('question_begin', None)
    question_end = data.get('question_end', None)
    max_new_token = data.get('max_new_token', 1024)
    num_choices = data.get('num_choices', 1)
    num_gpus_per_model = data.get('num_gpus_per_model', 1)
    num_gpus_total = data.get('num_gpus_total', 1)
    max_gpu_memory = data.get('max_gpu_memory', 70)
    dtype = str_to_torch_dtype(data.get('dtype', None))
    cache_dir = os.environ.get('CACHE_DIR', "/root/autodl-tmp/model")
    print("model_names:", model_names, "model_ids:", model_ids, "data_ids:", data_ids, "cache_dir:", cache_dir)
    failed = []
    if num_gpus_total // num_gpus_per_model > 1:
        import ray
        ray.init()
    
    try:
        start_time = get_start_time()
        outputs = []
        for data_id in data_ids:
            question_file = os.path.join(BASE_PATH, "llm_judge", "data", str(data_id), "question.jsonl")
            for model_name, model_id in zip(model_names, model_ids):
                model_name_saved = model_name.split('/')[-1]
                output_file = os.path.join(BASE_PATH, "llm_judge", "data", str(data_id), "model_answer", f"{model_name_saved}.jsonl")
                if is_non_empty_file(output_file):
                    print(
                        f"Skipping model_id {model_id} for data_id {data_id} as output file already exists and is non-empty.")
                else:
                    print("eval model:", model_name, model_id)
                    try:
                        run_eval(
                            ray=ray,
                            model_path=model_name, model_id=model_id, question_file=question_file,
                            question_begin=question_begin, question_end=question_end,
                            answer_file=output_file, max_new_token=max_new_token,
                            num_choices=num_choices, num_gpus_per_model=num_gpus_per_model,
                            num_gpus_total=num_gpus_total, max_gpu_memory=max_gpu_memory,
                            dtype=dtype, revision=revision, cache_dir=cache_dir
                        )
                    except AttributeError as e:
                        print("eval model error:", model_name, model_id)
                        print(e)
                        failed.append({"model_id": model_id, "reason": str(e)})
                        continue
                    except torch.cuda.OutOfMemoryError as e1:
                        print("eval model error:", model_name, model_id)
                        print(e1)
                        failed.append({"model_id": model_id, "reason": str(e1)})
                        continue
                temp = {"data_id": data_id,
                        "model_id": model_id, "model_name": model_name,
                        "output": output_file}
                outputs.append(temp)
        
        end_time = get_end_time()
        result = {
            "outputs": outputs,
            "model_names": model_names,
            "model_ids": model_ids,
            "data_ids": data_ids,
            "time_start": start_time,
            "time_end": end_time,
            "failed": failed
        }
        log_folder = os.path.join(BASE_PATH, "llm_judge", "log")
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, "eval_log.jsonl")
        print("log_path:", log_path)
        append_dict_to_jsonl(log_path, {request_id: result})
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5004)

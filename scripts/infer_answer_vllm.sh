#!/bin/bash

export PYTHONPATH="/root/autodl-tmp/software/FastChat/fastchat/llm_judge:$PYTHONPATH"

python fastchat/llm_judge/gen_model_answer.py --model-path /root/autodl-tmp/model/qwen/Qwen-7B-Chat --model-id qwen-7b-chat --bench-name moral_bench
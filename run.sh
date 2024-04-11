#!/bin/bash


read -p "input job name : " job_discribe
time_stamp=`date "+%m-%d-%H-%M"`
job_name="${job_discribe}-${time_stamp}"

echo "Job name: $job_name"

#group_name="nlp-0-szwg01-k8s-gpu-k40-4,idl-32g-0-yq1-k8s-gpu-v100-8,idl-0-yq01-k8s-gpu-p40-8"
#group_name="idl-16g-0-yq01-k8s-gpu-v100-8"
#group_name="idl-32g-0-yq01-k8s-gpu-v100-8"
#group_name="idl-0-yq01-k8s-gpu-p40-8"
#group_name="versionlab-40g-0-yq01-k8s-gpu-a100-8"
group_name="VisionLab-A100-Oversell"


job_version="pytorch-1.7.1"
#job_version="pytorch-1.4.0"
#start_cmd="bash env_run.sh && python train.py"
start_cmd="bash env_run.sh"
k8s_gpu_cards=4
k8s_trainers="1"
wall_time="100:00:00"
k8s_priority="high"
file_dir="."

#### modified
ak="c9e07c57f36e5d3cb3b0e1832f4a0547"
sk="f4a9a279cf6055958d7ca9d3c9164747"

paddlecloud job --ak ${ak} --sk ${sk} \
        train --job-name ${job_name} \
        --job-conf "config.ini" \
        --group-name ${group_name} \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --is-standalone 1

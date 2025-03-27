#!/bin/bash

set -e

DATA_DIR="data/benchmark"
ROBOT_CFG_PATH="content/configs/robot"
OUTPUT_DIR="logs/benchmark"

ROBOT_LIST=("franka" "iiwa" "kinova_gen3" "ur10e")
for ROBOT_NAME in "${ROBOT_LIST[@]}"; do
    echo "Processing robot ${ROBOT_NAME}"
    FILES=$(find $DATA_DIR -type f -name "*${ROBOT_NAME}*.pkl")
    for FILE in $FILES; do
        echo "Processing trajectory $FILE"
        python main.py \
            --log_level SUCCESS \
            --log_dir "${OUTPUT_DIR}" \
            --robot_cfg_path "${ROBOT_CFG_PATH}/${ROBOT_NAME}_w_base.yml" \
            --traj_data_path "${FILE}" # \
            # --disable_initial_guess \
            # --ret_all_steps
    done
done

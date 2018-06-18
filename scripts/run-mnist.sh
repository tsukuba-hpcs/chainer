#!/usr/bin/env bash
set -eu

# Usage:
#    run-mnist.sh [options]
#
# Options:
#    --train-script-path The path to Python training script.
#    --data-dir The path to the MNIST dataset root directory.


# Register all devices for which the MNIST training script should be tested.
devices=("native:0")
if [ -x "$(command -v nvcc)" ]; then
    devices+=("cuda:0")
fi

function get_loss() {
    loss="${1#*loss=}"
    echo "${loss%,*}"
}

function get_accuracy() {
    loss="${1#*accuracy=}"
    echo "${loss%,*}"
}

# Helper function to compare floats, i.e. the loss and the accuracy.
function compare_numbers() {
   awk 'BEGIN {print ("'$1'" < "'$2'")}'
}

train_script_path=""
data_dir=""

while [ $# -gt 0 ]; do
    o="$1"
    shift
    case "$o" in
        "--train-script-path")
            train_script_path="$1"
            shift
            ;;
        "--data-dir")
            data_dir="$1"
            shift
            ;;
        *)
            echo "$0: Unknown option: $o" >&2
            exit 1
    esac
done

for device in "${devices[@]}"
do
    # TODO(hvy): Train for longer when the performance of the native device is improved.
    IFS=$'\n'
    outputs=($(python "${train_script_path}" --data="${data_dir}" --device="${device}" --iteration=3 --batchsize=10 --eval-size=10 --eval-interval=iter))

    loss_begin="$(get_loss ${outputs[0]})"
    acc_begin="$(get_accuracy ${outputs[0]})"
    loss_end="$(get_loss ${outputs[-1]})"
    acc_end="$(get_accuracy ${outputs[-1]})"

    if [ "$(compare_numbers ${loss_begin} ${loss_end})" -eq 1 ]; then
        echo "Loss did not decrease on ${device}: ${loss_begin} -> ${loss_end}."
        exit 1
    fi

    if [ "$(compare_numbers ${acc_end} ${acc_begin})" -eq 1 ]; then
        echo "Accuracy did not increase on ${device}: ${acc_begin} -> ${acc_end}."
        exit 1
    fi

    echo "Successfully trained ${device}."
done

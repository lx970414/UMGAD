#!/usr/bin/env bash
set -e
export PYTHONPATH=$(pwd)
python train.py "$@"

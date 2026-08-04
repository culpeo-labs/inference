#!/bin/bash
EXECUTABLE=$1
CHECKPOINT=$2
PROMPT=$3

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
python3 $SCRIPT_DIR/encode.py "$CHECKPOINT" "$PROMPT" | $EXECUTABLE $CHECKPOINT | python3 $SCRIPT_DIR/decode.py "$CHECKPOINT"
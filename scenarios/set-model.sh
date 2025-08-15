#!/bin/bash

if [[ "$1" != "" ]]; then
    NewModelId="$1"
else
    echo "Usage: $0 <model_id>"
    echo "Please provide a model ID to set."
    exit 1
fi

# Check if the model ID is valid
if [[ ! "$NewModelId" =~ ^[\w\-_]+\/[\w\-_]+$ ]]; then
    echo "Invalid model ID format. Please use alphanumeric characters, underscores, or hyphens."
    exit 1
fi

# call utils.update_used_model python func with model_id
python3 -c "
import sys
from utils import update_used_model
if __name__ == '__main__':
    model_id = sys.argv[1]
    update_used_model(model_id)
" "$NewModelId"
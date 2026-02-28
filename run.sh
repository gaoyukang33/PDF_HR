CUDA_ID=$1
ARG_FILE=$2
CUDA_VISIBLE_DEVICES="$CUDA_ID" python mimickit/run.py --arg_file "$ARG_FILE" --visualize false 
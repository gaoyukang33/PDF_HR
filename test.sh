CUDA_ID=$1
MODEL_FOLDER=$2
MODEL_FILE=$3
CUDA_VISIBLE_DEVICES="$CUDA_ID" python mimickit/test_viser.py --model_folder "$MODEL_FOLDER" --model_file "$MODEL_FILE"
for dataset in "dior-rsvg" "geochat" "rsvg" "vrsbench"
do
    GPUS=8 bash evaluate.sh "models/ViTP_InternVL_1B_rs" $dataset --dynamic 
done


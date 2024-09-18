# # 1. Check and make the following folders are deleted:   ./data/split_ss_test   ./results   ./xml_results
# # 2. Change ./tools/data/dota/split/split_configs/ms_test.json line 4 to /input_path
# # 3. Run:
# pip install -v -e .
# python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_test.json
# python tools/test.py ./lsk_s_fpn_1x_for_test.py ./data/lsk_s_fpn_1x_epoch_12.pth --format-only --eval-options submission_dir=./results 
# # ./data/input_path is a fake test image folder for testing purposes
# python tools/data/fair/dota_to_fair.py ./results ./xml_results ./data/input_path # Change ./data/input_path to /input_path 

./tools/dist_train.sh ./local_configs/tianzhibei_multidata_convnext_b_sardetdota_pretrained_orcnn_frcnn.py 8

# scancel 

export input_dir=/mnt/bn/haotongdata/Datasets/scannetpp/data/5f99900f09/merge
export matcher=exhaustive_matcher
python3 scripts/colmap.py --input_dir $input_dir --matcher $matcher
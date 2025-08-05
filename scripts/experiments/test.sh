#!/bin/zsh
mlx_submit_v2() {
  if [[ -e scripts/mlx_submit/run_v2.py ]]; then
    python3 scripts/mlx_submit/run_v2.py "$@"
  else
    echo "错误：scripts/mlx_submit/run_v2.py 不存在，请确认路径是否正确。"
  fi
}
# 61adeff7d5
# 281ba69af1
# 0e75f3c4d9
# 1f7cbbdde1
# 95d525fbfd
# 69e5939669
# 324d07a5b3
# 076c822ecc
# 8e6ff28354
# e91722b5a3
# 47b37eb6f9
# 6855e1ac32
# a29cccc784
# 38d58a7a31
# 6cc2231b9c
# 67d702f2e8
# a003a6585e
# 0a184cf634
# a8bf42d646
# 104acbf7d2
# 4c5c60fa76
# 32280ecbca
# cbd4b3055e
# 49a82360aa
# 8be0cd3817
# 2970e95b65
# bde1e479ad
# 290ef3f2c9
# bd7375297e
# 7f4d173c9c
# bc400d86e1
# a05ee63164
# 89214f3ca0
# c4c04e6d6c
# 6ee2fc1070
scenes=(
    "27dd4da69e"
    "39e6ee46df"
    "9f79564dbf"
    "6f1848d1e3"
    "ac48a9b736"
    "578511c8a9"
    "8133208cb6"
    "3db0a1c8f3"
    "61adeff7d5"
    "281ba69af1"
    "0e75f3c4d9"
    "1f7cbbdde1"
    "95d525fbfd"
    "69e5939669"
    "324d07a5b3"
    "076c822ecc"
    "8e6ff28354"
    "e91722b5a3"
    "47b37eb6f9"
    "6855e1ac32"
    "a29cccc784"
    "38d58a7a31"
    "6cc2231b9c"
    "67d702f2e8"
    "a003a6585e"
    "0a184cf634"
    "a8bf42d646"
    "104acbf7d2"
    "4c5c60fa76"
    "32280ecbca"
    "cbd4b3055e"
    "49a82360aa"
    "8be0cd3817"
    "2970e95b65"
    "bde1e479ad"
    "290ef3f2c9"
    "bd7375297e"
    "7f4d173c9c"
    "bc400d86e1"
    "a05ee63164"
    "89214f3ca0"
    "c4c04e6d6c"
    "6ee2fc1070"
)
# 数组长度
len=${#scenes[@]}
submit_job() {
    local scenes=("$@")
    local job_name="zipnerf"
    for scene in "${scenes[@]}"; do
        job_name+="_${scene:0:5}"
    done
    local command="mlx_submit_v2 --job_name $job_name --num_mem 140 bash scripts/experiments/scannetpp_zipnerf_batch.sh ${scenes[*]}"
    echo "Executing: $command"
    $command
}
for ((i = 0; i < len; i += 6)); do
    batch=("${scenes[@]:i:6}")
    submit_job "${batch[@]}"
done

#!/bin/bash
scenes=(
    "7b6477cb95" "c50d2d1d42" "cc5237fd77" "acd95847c5"
    "fb5a96b1a2" "a24f64f7fb" "1ada7a0617" "5eb31827b7"
    "3e8bba0176" "3f15a9266d" "21d970d8de" "5748ce6f01"
    "c4c04e6d6c" "7831862f02" "bde1e479ad" "38d58a7a31"
    "5ee7c22ba0" "f9f95681fd" "3864514494" "40aec5fffa"
    "13c3e046d7" "e398684d27" "a8bf42d646" "45b0dac5e3"
    "31a2c91c43" "e7af285f7d" "286b55a2bf" "7bc286c1b6"
    "f3685d06a9" "b0a08200c9" "825d228aec" "a980334473"
    "f2dc06b1d2" "5942004064" "25f3b7a318" "bcd2436daf"
    "f3d64c30f8" "0d2ee665be" "3db0a1c8f3" "ac48a9b736"
    "c5439f4607" "578511c8a9" "d755b3d9d8" "99fa5c25e1"
    "09c1414f1b" "5f99900f09" "9071e139d9" "6115eddb86"
    "27dd4da69e" "c49a8c6cff"
)
# 每次处理8个场景
batch_size=4
for ((i=0; i<${#scenes[@]}; i+=batch_size)); do
    # 为当前批次的每个场景启动一个后台任务
    echo "Processing batch starting from ${i}"
    echo "Processing scenes ${scenes[@]:i:batch_size}"
    for ((j=0; j<batch_size && (i+j)<${#scenes[@]}; j++)); do
        scene=${scenes[i+j]}
        # echo ${scene} ${j}
        export CUDA_VISIBLE_DEVICES=${j}
        python3 scripts/scannetpp/run_fuse.py --scene ${scene} --fusion_type pred &
    done

    # 等待当前批次的所有任务完成
    wait
    echo "Batch ${i} done."
    echo "Processing next batch."
done
echo "所有命令已完成执行。"
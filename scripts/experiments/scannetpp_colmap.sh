cd $ICCV_CODE_PATH
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <value1> <value2>"
    exit 1
fi
export batch_id="$1"
export batch_size="$2"
echo ${batch_id}
echo ${batch_size}
cmd1="python3 scripts/scannetpp/run_process_mapper.py --batch_id ${batch_id} --batch_size ${batch_size} --num_processes 10 "
cmd2="python3 /mnt/bn/haotongdata/home/linhaotong/envs/idle2.py"
$cmd1 &
cmd1_pid=$!
$cmd2 &
cmd2_pid=$!
echo $cmd1_pid
echo $cmd2_pid
wait $cmd1_pid
echo "cmd1 finished"
kill -9 $cmd2_pid

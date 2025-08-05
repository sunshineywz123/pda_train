export workspace=/mnt/bn/haotongdata/home/linhaotong/workspaces/pl_htcode
export output_file=${workspace}/processed_datasets/ARKitScenes/upsampling/bilinear_upsampling_metrics.txt
python3 scripts/arkitscenes/report_upsampling_metric.py --conf_level 0 --up_scale 1 >> $output_file
python3 scripts/arkitscenes/report_upsampling_metric.py --conf_level 1 --up_scale 1 >> $output_file
python3 scripts/arkitscenes/report_upsampling_metric.py --conf_level 2 --up_scale 1 >> $output_file
python3 scripts/arkitscenes/report_upsampling_metric.py --conf_level -1 --up_scale 1 >> $output_file
python3 scripts/arkitscenes/report_upsampling_metric.py --up_scale 2 >> $output_file
python3 scripts/arkitscenes/report_upsampling_metric.py --up_scale 4 >> $output_file
python3 scripts/arkitscenes/report_upsampling_metric.py --up_scale 8 >> $output_file
echo "Bilinear upsampling metrics are saved in $output_file"
# scenes=(
# "1ada7a0617"
# "1ae9e5d2a6"
# "1b9692f0c7"
# "1d003b07bd"
# )
# scenes从命令行参数读取
# scenes=("$@")
cd $ICCV_CODE_PATH
scenes=("$@")
for scene in "${scenes[@]}"
do
  echo $scene
  export scene=$scene
  bash scripts/experiments/scannetpp_zipnerf.sh ${scene}
  cd $ICCV_CODE_PATH
done
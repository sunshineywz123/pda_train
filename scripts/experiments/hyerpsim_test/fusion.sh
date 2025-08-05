echo "Running fusion for scene ${scene} and camera ${cam}"
python3 scripts/hypersim/run_fuse.py --scene ${scene} --cam ${cam} --type lowres
python3 scripts/hypersim/run_fuse.py --scene ${scene} --cam ${cam} --type pred
python3 scripts/hypersim/run_fuse.py --scene ${scene} --cam ${cam} --type gt
echo "Fusion done for scene ${scene} and camera ${cam}"

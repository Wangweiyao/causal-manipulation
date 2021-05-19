run_name=hard-anneal
rm -r results/$run_name

CUDA_VISIBLE_DEVICES=1 python train.py \
    --experience-replay ../data_collection/data/data_block_env/experience.pth \
    --epochs 1000 \
    --batch-size 11 \
    --chunk-size 10 \
    --frame-skip 1 \
    --print-freq 500 \
    --phase-simplify-summary False \
    --ckpt-dir results/${run_name}/model \
    --summary-dir results/${run_name}/summary


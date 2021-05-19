run_name=hard-finetune-only-kl

CUDA_VISIBLE_DEVICES=1 python finetune.py \
    --experience-replay ../data_collection/data/data_block_env/experience.pth \
    --epochs 1000 \
    --batch-size 11 \
    --chunk-size 10 \
    --frame-skip 1 \
    --print-freq 500 \
    --phase-simplify-summary False \
    --ckpt-dir results/${run_name}/model \
    --summary-dir results/${run_name}/summary \
    --last-ckpt results/hard/model/ckpt_epoch_55000.pth


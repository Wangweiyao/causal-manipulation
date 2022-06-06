
CUDA_VISIBLE_DEVICES=1 python train_gap.py \
    --id  gap_block \
    --env GapBlock-v0 \
    --experience-replay ../data_collection/data/data_block_env/experience.pth \
    --hidden-size 256 \
    --batch-size 32 \
    --chunk-size 30 \
    --trainsteps 750000 --checkpoint-interval 100000
    #--models /home/weiyao/robot/gap/results/trained_gap_block/models_160000.pths

#CUDA_VISIBLE_DEVICES=1 python train_svg.py \
#    --id svg_block \
#    --env GapBlock-v0 \
#   --experience-replay ../data_collection/data/data_block_env/experience.pth \
#   --hidden-size 256 \
#    --batch-size 32 \
#    --chunk-size 30 \
#    --trainsteps 750000 --checkpoint-interval 100000
    #--models /home/weiyao/robot/gap/results/trained_gap_block/models_160000.pths

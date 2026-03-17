echo ">>> Step 1: Cropping offline training data..."
python ./DataLoader/crop_train_data.py \
    --input ./dataset/train/train \
    --output ./dataset/train/crop_size256_stride128 \
    --size 256 \
    --stride 128

python ./DataLoader/crop_train_data.py \
    --input ./dataset/train/train \
    --output ./dataset/train/crop_size512_stride256 \
    --size 512 \
    --stride 256

python ./DataLoader/crop_train_data.py \
    --input ./dataset/train/train \
    --output ./dataset/train/crop_size768_stride384 \
    --size 768 \
    --stride 384
echo "Dataset preparation complete!"

echo ">>> Step 2: Starting DDP training..."
accelerate launch --multi_gpu --num_processes 4 train.py \
    --train_data ./dataset/train/crop_size256_stride128 \
    --batch_size 4 \
    --epochs 20 \
    --restart

accelerate launch --multi_gpu --num_processes 4 train.py \
    --train_data ./dataset/train/crop_size512_stride256 \
    --batch_size 1 \
    --epochs 30 

accelerate launch --multi_gpu --num_processes 4 train.py \
    --train_data ./dataset/train/crop_size768_stride384 \
    --batch_size 1 \
    --epochs 70 \
    --mixed_precision "bf16" \
    --use_checkpoint 
echo "Training phases complete!"

echo ">>> Step 3: Starting evaluating..."
python ./eval.py \
    --ckpt_dir ./checkpoint \
    --eval_dir ./output_eval_phase2 \
    --test_root ./dataset/testdata_phase2 \
    --name model_best.pth.tar

python ./eval.py \
    --ckpt_dir ./checkpoint \
    --eval_dir ./output_eval_phase3 \
    --test_root ./dataset/testdata_phase3 \
    --name model_best.pth.tar
echo "Evaluating phases complete!"
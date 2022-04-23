export CUDA_VISIBLE_DEVICES=1
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_lr1e-4" \
 --train_file "DuSinc/query/train.csv" \
 --validation_file "DuSinc/query/dev.csv" \
 --test_file "DuSinc/query/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --scenario_name "batch_size 8,baseline_input,lr:1e-4" \
 --num_train_epochs 10 \
 --learning_rate 1e-4 \
 --context_pos 1 > nohup/query_1e-4.out 2>&1 &

export CUDA_VISIBLE_DEVICES=0
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_lr5e-5" \
 --train_file "DuSinc/query/train.csv" \
 --validation_file "DuSinc/query/dev.csv" \
 --test_file "DuSinc/query/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --scenario_name "batch_size 8,baseline_input,lr:5e-5" \
 --num_train_epochs 10 \
 --learning_rate 5e-5 \
 --context_pos 1  > nohup/query_5e-5.out 2>&1 &

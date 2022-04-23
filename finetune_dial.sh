export CUDA_VISIBLE_DEVICES=3
accelerate launch response_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/response_baseline_lr3e-5" \
 --train_file "DuSinc/response/train.csv" \
 --validation_file "DuSinc/response/dev.csv" \
 --test_file "DuSinc/response/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name response_gen \
 --scenario_name "batch_size 8, knowledge+location+context, lr:3e-5" \
 --num_train_epochs 10 \
 --learning_rate 3e-5  \
 --context_pos 2 > nohup/nohup_gen_baseline_3e-5.out 2>&1 &

export CUDA_VISIBLE_DEVICES=2
accelerate launch response_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/response_baseline_lr2e-5" \
 --train_file "DuSinc/response/train.csv" \
 --validation_file "DuSinc/response/dev.csv" \
 --test_file "DuSinc/response/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name response_gen \
 --scenario_name "batch_size 8, knowledge+location+context, lr:2e-5" \
 --num_train_epochs 10 \
 --learning_rate 2e-5  \
 --context_pos 2 > nohup/nohup_gen_baseline_2e-5.out 2>&1 &
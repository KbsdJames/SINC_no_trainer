export CUDA_VISIBLE_DEVICES=3
accelerate launch response_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/response_baseline" \
 --train_file "DuSinc/response/train.csv" \
 --validation_file "DuSinc/response/dev.csv" \
 --test_file "DuSinc/response/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name response_gen \
 --scenario_name "batch_size4*5, knowledge+location+context, lr:5e-5" \
 --num_train_epochs 10 \
 --learning_rate 5e-5  \
 --context_pos 2 #> nohup/nohup_gen_baseline_5e-5.out 2>&1
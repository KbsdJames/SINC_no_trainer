accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_baseline" \
 --train_file "DuSinc/dialogue/train.csv" \
 --validation_file "DuSinc/dialogue/dev.csv" \
 --test_file "DuSinc/dialogue/test.csv" \
 --mode "train" \
 --per_device_batch_size 5 \
 --experiment_name query_gen \
 --scenario_name "batch_size4*5,baseline_input" \
 --num_train_epochs 10  > nohup.out 2>&1 &


export CUDA_VISIBLE_DEVICES=1
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_trunc" \
 --train_file "DuSinc/query/train.csv" \
 --validation_file "DuSinc/query/dev.csv" \
 --test_file "DuSinc/query/test.csv" \
 --mode "train" \
 --per_device_batch_size 5 \
 --experiment_name query_gen \
 --scenario_name "batch_size4*5,baseline_input trunc_512" \
 --num_train_epochs 10 \
 --context_pos 1 > nohup_query.out 2>&1

export CUDA_VISIBLE_DEVICES=0
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_trunc_topic" \
 --train_file "DuSinc/query_topic/train.csv" \
 --validation_file "DuSinc/query_topic/dev.csv" \
 --test_file "DuSinc/query_topic/test.csv" \
 --mode "train" \
 --per_device_batch_size 5 \
 --experiment_name query_gen \
 --scenario_name "batch_size4*5,baseline_input trunc_512 topic" \
 --num_train_epochs 10 \
 --context_pos 2  > nohup_query_topic.out 2>&1

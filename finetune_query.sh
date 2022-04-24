# 之前有梯度爆炸的情况
export CUDA_VISIBLE_DEVICES=1
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_lr5e-5_warm1000" \
 --train_file "DuSinc/query/train.csv" \
 --validation_file "DuSinc/query/dev.csv" \
 --test_file "DuSinc/query/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --scenario_name "batch_size 8,baseline_input,lr5e-5,warm1000" \
 --num_train_epochs 10 \
 --learning_rate 5e-5 \
 --num_warmup_steps 1000 \
 --context_pos 1 > nohup/query_5e-5_warm1000.out 2>&1 &

# 训基本情况用于比较
export CUDA_VISIBLE_DEVICES=0
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_lr5e-5_warm1000" \
 --train_file "DuSinc/query/train.csv" \
 --validation_file "DuSinc/query/dev.csv" \
 --test_file "DuSinc/query/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --scenario_name "batch_size 8,baseline_input,lr:5e-5,warm1000" \
 --num_train_epochs 10 \
 --learning_rate 5e-5 \
 --num_warmup_steps 1000 \
 --context_pos 2 > nohup/query_topic_5e-5_warm1000.out 2>&1 &

# 梯度爆炸是否会有改善
export CUDA_VISIBLE_DEVICES=2
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_lr1e-4_weight_decay0.01_warmup1000" \
 --train_file "DuSinc/query/train.csv" \
 --validation_file "DuSinc/query/dev.csv" \
 --test_file "DuSinc/query/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --weight_decay 0.01 \
 --scenario_name "batch_size 8,baseline_input,lr1e-4,weight_decay0.01_warmup1000" \
 --num_train_epochs 10 \
 --learning_rate 1e-4 \
 --num_warmup_steps 1000 \
 --context_pos 1 > nohup/query_1e-4_weight_decay0.01_warmup1000.out 2>&1 &


# 最佳结果
 export CUDA_VISIBLE_DEVICES=3
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_topic_lr5e-5_weight_decay0.01,warm1000" \
 --train_file "DuSinc/query_topic/train.csv" \
 --validation_file "DuSinc/query_topic/dev.csv" \
 --test_file "DuSinc/query_topic/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --weight_decay 0.01 \
 --scenario_name "batch_size 8,add_topic,lr5e-5,weight_decay0.01,warm1000" \
 --num_train_epochs 10 \
 --learning_rate 5e-5 \
 --num_warmup_steps 1000 \
 --context_pos 2 > nohup/query_topic_5e-5_weight_decay0.01_warmup1000.out 2>&1 &
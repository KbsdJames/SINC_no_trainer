export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch query_gen_notrainer.py \
 --model_path fnlp/cpt-base \
 --output_dir "output/query_topic_lr5e-5_weight_decay0.01_topic_batch32" \
 --train_file "DuSinc/query_topic/train.csv" \
 --validation_file "DuSinc/query_topic/dev.csv" \
 --test_file "DuSinc/query_topic/test.csv" \
 --mode "train" \
 --per_device_batch_size 8 \
 --experiment_name query_gen \
 --weight_decay 0.01 \
 --scenario_name "batch_size 8,add_topic,lr5e-5,weight_decay0.01_topic_batch32" \
 --num_train_epochs 10 \
 --learning_rate 5e-5 \
 --context_pos 2 > nohup/query_topic_5e-5_weight_decay0.01_topic_batch32.out 2>&1 &
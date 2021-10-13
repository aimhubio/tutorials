### Excellent reproducibility: master the experiment in your hands

This folder contains a sample training code that logs / tracks all the points made in the blogpost.

#### Run

```shell
pip install git+https://github.com/huggingface/transformers

pip install -r requirements.txt

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name 'mnli' \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir 'mnli_log'
```




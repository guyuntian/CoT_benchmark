### Training
To train minGPT on direct dataset with **single-node training**, run the following on 1 node with 4 GPUs:
```
torchrun --nproc_per_node=4 train.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --output_dir ${OUTPUT_DIR}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
 --weight_decay 0.01\
 --learning_rate 1e-4\
 --drop 0.1\
 --batch_size 128\
 --epoch 100\
 --warmup 5\
 --dmodel 256\
 --head 4\
 --num_layer ${LAYER}
```
- Here `folder` should be one of the followings: arithmetic, equation, LIS or ED.
- `maxdata` always equals to `maxlen` in direct dataset.
- The effective batch size is 128 (`batch_size` per gpu) * 3 (gpus per node) = 512.

The numbers of MAXLEN, VOCAB_SIZE and NUM_RANGE are listed in the following table.

| Dataset      | MAXLEN | VOCAB_SIZE | NUM_RANGE |
| ----------- | ----------- | ----------- | ----------- |
| Arithmetic_len4  | 19     | 21 | 11 |
| Arithmetic_len5  | 23     | 21 | 11 |
| Arithmetic_len6  | 27     | 21 | 11 |
| Equation_len3  | 46     | 22 | 11 |
| Equation_len4  | 73     | 22 | 11 |
| Equation_len5  | 106     | 22 | 11 |
| LIS_len50  | 54     | 203 | 200 |
| LIS_len80  | 84     | 233 | 230 |
| LIS_len100  | 104     | 253 | 250 |
| ED_len12  | 31     | 91 | 60 |
| ED_len16  | 39     | 91 | 60 |
| ED_len20  | 47     | 91 | 60 |

To train minGPT on CoT dataset, run the following on 1 node with 4 GPUs:
```
torchrun --nproc_per_node=4 train.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --output_dir ${OUTPUT_DIR}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
 --weight_decay 0.01\
 --learning_rate 1e-4\
 --drop 0.1\
 --batch_size 128\
 --epoch 100\
 --warmup 5\
 --dmodel 256\
 --head 4\
 --num_layer 3\
 --chain
```
- `maxdata` always equals to `maxlen` in CoT dataset.

The numbers of MAXLEN, VOCAB_SIZE and NUM_RANGE are listed in the following table.

| Dataset      | MAXLEN | VOCAB_SIZE | NUM_RANGE |
| ----------- | ----------- | ----------- | ----------- |
| Arithmetic_len4  | 43     | 21 | 11 |
| Arithmetic_len5  | 63     | 21 | 11 |
| Arithmetic_len6  | 87     | 21 | 11 |
| Equation_len3  | 91     | 22 | 11 |
| Equation_len4  | 181     | 22 | 11 |
| Equation_len5  | 316     | 22 | 11 |
| LIS_len50  | 105     | 203 | 200 |
| LIS_len80  | 165     | 233 | 230 |
| LIS_len100  | 205     | 253 | 250 |
| ED_len12  | 212     | 91 | 60 |
| ED_len14  | 344     | 91 | 60 |
| ED_len16  | 508     | 91 | 60 |

To train minGPT on CoT dataset with [alibi relative positional embeddings](https://arxiv.org/abs/2108.12409), run the following on 1 node with 4 GPUs:
```
torchrun --nproc_per_node=4 train.py\
 --file ${DATA_DIR}\
 --folder arithmetic\
 --output_dir ${OUTPUT_DIR}\
 --maxlen 1000\
 --maxdata 435\
 --vocab 21\
 --num_range 11\
 --weight_decay 0.01\
 --learning_rate 1e-4\
 --drop 0.1\
 --batch_size 128\
 --epoch 100\
 --warmup 5\
 --dmodel 256\
 --head 4\
 --num_layer 3\
 --chain\
 --rpe
```

### Testing
To test minGPT performance on direct dataset, run the following on 1 node with 4 GPUs:
```
torchrun --nproc_per_node=4 test.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
 --batch_size 128\
 --dmodel 256\
 --head 4\
 --num_layer ${LAYER}\
 --model_path ${MODEL_PATH}
```
The number of MAXLEN, MAXDATA, VOCAB_SIZE and NUM_RANGE should be consistent with the training config.

To test minGPT performance on CoT dataset, run the following on 1 node with 4 GPUs:
```
torchrun --nproc_per_node=4 test.py\
 --file ${DATA_DIR}\
 --folder ${TASK}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
 --batch_size 128\
 --dmodel 256\
 --head 4\
 --num_layer 3\
 --model_path ${MODEL_PATH}\
 --chain
```

To test minGPT on CoT dataset with alibi, run the following on 1 node with 4 GPUs:
```
torchrun --nproc_per_node=4 test.py\
 --file ${DATA_DIR}\
 --folder arithmetic\
 --maxlen 1000\
 --maxdata ${MAXDATA}\
 --vocab 21\
 --num_range 11\
 --batch_size 128\
 --dmodel 256\
 --head 4\
 --num_layer 3\
 --chain\
 --rpe\
 --model_path ${MODEL_PATH}
```
The numbers of MAXDATA are listed in the following table.
| Number of Operators      | MAXDATA | 
| ----------- | ----------- | 
| 16  | 487     |
| 17  | 553     | 
| 18  | 607     | 

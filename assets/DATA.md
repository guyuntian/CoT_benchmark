## Data Generation
To get arithmetic expression dataset, run the following command:
```
python3 arithmetic/data.py \
    --file ${DATA_DIR} \
    --length ${NUMBER_OF_OPERATORS} \
    --train_size 1e6 \
    --test_size 1e5\
    --number_range 11\
    --under
```
- Here `number_range` specifies the number field (should be a prime).
- `under` means there is a part of training data whose number of operators is under ${NUMBER_OF_OPERATORS}.

Script for linear equation dataset:
```
python3 equation/data.py \
    --file ${DATA_DIR} \
    --length ${NUMBER_OF_VARIABLES} \
    --train_size 1e6 \
    --test_size 1e5\
    --number_range 11
```

Script for longest increasing subsequence dataset:
```
python3 LIS/data.py \
    --file ${DATA_DIR} \
    --length ${LEN_INPUTS} \
    --train_size 1e6 \
    --test_size 1e5\
    --number_range ${NUM_RANGE}
```
- In our experiment, we set `number_range` to `length` + 150.

Script for edit distance dataset:
```
python3 ED/data.py \
    --file ${DATA_DIR} \
    --length ${LEN_OF_FIRST_STRING} \
    --train_size 1e6 \
    --test_size 1e5\
    --using 8
```
- Here `using` + 2 = the max size of working vocabulary.
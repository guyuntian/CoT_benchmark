import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='data')

parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--length', type=int, default=40)
parser.add_argument('--train_size', type=float, default=1e6)
parser.add_argument('--test_size', type=float, default=1e5)
parser.add_argument('--number_range', type=int, default=150)

args = parser.parse_args()
np.random.seed(2023)

integer = np.arange(args.number_range)
num_start = args.length + 1

def get_seq(length):
    sub_seq = np.random.randint(3) + 1
    total = np.random.randint(length - 2) + 3
    if sub_seq == 1:
        increasing = [total]
    elif sub_seq == 2:
        tmp = np.random.randint(total // 2 + 1) + 1
        increasing = [tmp, total - tmp]
    else:
        tmp1 = np.random.randint(total // 3 + 1) + 1
        tmp2 = np.random.randint((total - tmp1) // 2 + 1) + 1
        increasing = [tmp1, tmp2, total - tmp1 - tmp2]
    numbers = np.concatenate([np.sort(np.random.choice(args.number_range - num_start,\
     increasing[i], replace=False)) + num_start for i in range(sub_seq)], axis=0)
    places = np.sort(np.random.choice(length, total, replace=False))
    seq = np.random.randint(num_start, high=args.number_range, size=length)
    seq[places] = numbers
    return seq

def solve(lst):
    length = len(lst)
    cot = np.ones(length, dtype=np.int32)
    for l in range(length):
        for i in range(l):
            if lst[l] > lst[i]:
                cot[l] = max(cot[i] + 1, cot[l])
    return cot

train_set = set()
max_len = 0
max_history = 0

while len(train_set) < args.train_size:
    lst = get_seq(args.length)
    cot = solve(lst)
    final = lst.tolist() + ["<sep>"] + cot.tolist() + ["<sep>"]
    final.append(np.max(cot))
    train_set.add(tuple(final))
    max_len = max(max_len, args.length + 2)
    max_history = max(max_history, len(final))

test_set = set()
while len(test_set) < args.test_size:
    lst = get_seq(args.length)
    cot = solve(lst)
    final = lst.tolist() + ["<sep>"] + cot.tolist() + ["<sep>"]
    final.append(np.max(cot))
    if tuple(final) not in train_set:
        test_set.add(tuple(final))
    max_len = max(max_len, args.length + 2)
    max_history = max(max_history, len(final))

os.makedirs(f"{args.file}", exist_ok=True)
decoder = f"{args.file}/decoder"
chain = f"{args.file}/chain"
os.makedirs(decoder, exist_ok=True)
os.makedirs(chain, exist_ok=True)

with open(f"{decoder}/train_data.txt", 'w') as f1:
    for lst in train_set:
        for i in lst:
            print(i, end=' ', file=f1)
            if i == "<sep>":
                break
        print(lst[-1], file=f1)

with open(f"{decoder}/test_data.txt", 'w') as f1:
    for lst in test_set:
        for i in lst:
            print(i, end=' ', file=f1)
            if i == "<sep>":
                break
        print(lst[-1], file=f1)


with open(f"{chain}/train_data.txt", 'w') as f1:
    for lst in train_set:
        for i in lst:
            print(i, end=' ', file=f1)
        print("", file=f1)

with open(f"{chain}/test_data.txt", 'w') as f1:
    for lst in test_set:
        for i in lst:
            print(i, end=' ', file=f1)
        print("", file=f1)

print(f"max direct len:{max_len}")
print(f"max cot len:{max_history}")
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='data')

parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--train_size', type=float, default=1e6)
parser.add_argument('--test_size', type=float, default=1e5)
parser.add_argument('--using', type=int, default=8)

args = parser.parse_args()
np.random.seed(2023)

alphabet = [i for i in "abcdefghijklmnopqrstuvwxyz"]

def get_seq(diff):
    using = np.random.randint(args.using) + 3
    available = np.random.choice(alphabet, using, replace=False)
    str1 = np.random.randint(using, size=args.length)
    str1 = [available[i] for i in str1]
    if np.random.rand() < 0.4:
        length = np.random.randint(args.length - 3, args.length + 3)
        str2 = np.random.randint(using, size=length)
        str2 = [available[i] for i in str2]
    else:
        str2 = str1[:]
        for _ in range(diff):
            a = np.random.randint(3)
            if a == 0 and len(str2) > 2:
                p = np.random.randint(len(str2))
                str2 = str2[:p] + str2[p+1:]
            elif a == 1:
                p = np.random.randint(len(str2))
                str2 = str2[:p] + [np.random.choice(available)] + str2[p+1:]
            else:
                p = np.random.randint(len(str2) + 1)
                str2 = str2[:p] + [np.random.choice(available)] + str2[p:]
    if str1 == str2 or len(str2) >= args.length + 3 or len(str2) < args.length - 3:
        return get_seq(diff)
    if len(str1) > len(str2):
        return str2, str1
    return str1, str2

def solve(str1, str2):
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1,len(str1)+1):
        for j in range(1,len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 3
            matrix[i][j] = min(matrix[i-1][j]+2,matrix[i][j-1]+2,matrix[i-1][j-1]+d)
    return matrix

train_set = set()
max_len = 0
max_history = 0

while len(train_set) < args.train_size:
    str1, str2 = get_seq(np.random.randint(args.length) + 1)
    matrix = solve(str1, str2)
    final = str1 + ["|"] + str2 + ["<sep>"]
    for row in range(1, len(matrix)):
        final = final + matrix[row][1:] + [","]
    final.append("<sep>")
    final.append(matrix[-1][-1])
    train_set.add(tuple(final))
    max_history = max(max_history, len(final))
    max_len = max(max_len, len(str1) + len(str2) + 3)

test_set = set()
while len(test_set) < args.test_size:
    str1, str2 = get_seq(np.random.randint(args.length) + 1)
    matrix = solve(str1, str2)
    final = str1 + ["|"] + str2 + ["<sep>"]
    for row in range(1, len(matrix)):
        final = final + matrix[row][1:] + [","]
    final.append("<sep>")
    final.append(matrix[-1][-1])
    if tuple(final) not in train_set:
        test_set.add(tuple(final))
    max_history = max(max_history, len(final))
    max_len = max(max_len, len(str1) + len(str2) + 3)


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
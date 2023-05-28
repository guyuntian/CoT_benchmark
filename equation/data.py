import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='data')

parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--length', type=int, default=3)
parser.add_argument('--train_size', type=float, default=1e6)
parser.add_argument('--test_size', type=float, default=1e5)
parser.add_argument('--number_range', type=int, default=11)

args = parser.parse_args()
np.random.seed(2023)

rang = args.number_range
nums = [i for i in range(rang)]
mul = np.arange(rang).reshape(-1,1) * np.arange(rang).reshape(1, -1)
mul = mul % rang

div = np.zeros((rang, rang), dtype=np.int32)
for i in range(rang):
    pos = np.where(mul==i)
    div[i, pos[0]] = pos[1]

dic = dict()
variable = "abcde"
for i in range(len(variable)):
    dic[i] = variable[i]

def output(X, step):  
    ans = []
    for x in X:
        for j in range(x.shape[0]):
            if j == x.shape[0] - 1:
                ans.pop()
                ans.append("=")
                ans.append(x[j])
                continue
            if x[j] == 0 and j < step:
                continue
            if x[j] == 1 and j < step:
                ans.append(dic[j])
            else:
                ans.append(x[j])
                ans.append(dic[j])
            ans.append("+")
        ans.append(",")
    return ans

def get(length):
    matrix = np.random.randint(0, rang, size=(length, length + 1))
    history = []
    history.append(matrix + 0)
    for j in range(length):
        if np.sum(matrix[j:, j]) == 0:
            break
        pos = np.argmax(np.where(matrix[j:,j] != 0, 1, 0)) + j
        if pos != j:
            matrix[[pos, j]] = matrix[[j, pos]]

        division = matrix[j, j]
        if division != 1:
            for k in range(length + 1):
                matrix[j, k] = div[matrix[j, k], division]
        
        if np.sum(matrix[:,j]) != matrix[j, j]:
            for k in range(length):
                if j == k:
                    continue
                matrix[k] = (matrix[k] - matrix[k, j] * matrix[j]) % rang
        history.append(matrix + 0)

    if np.sum(matrix[j:, j]) != 0 and len(history) != 1:
        ans_seq = []
        step = 0
        for matri in history:
            ans_seq = ans_seq + output(matri, step)
            step += 1
            ans_seq[-1] = "<sep>"
        ans_seq.pop()
        return ans_seq
    return get(length)

train_set = set()
while len(train_set) < args.train_size:
    history = get(args.length)
    train_set.add(tuple(history))

test_set = set()
while len(test_set) < args.test_size:
    history = get(args.length)
    tmp = tuple(history)
    if tmp not in train_set:
        test_set.add(tmp)


decoder = f"{args.file}/decoder"
chain = f"{args.file}/chain"
os.makedirs(f"{args.file}", exist_ok=True)
os.makedirs(decoder, exist_ok=True)
os.makedirs(chain, exist_ok=True)

maxlen = 0
with open(f"{decoder}/train_data.txt", 'w') as f1:
    for history in train_set:
        this_len = 0
        for i in history:
            print(i, end=' ', file=f1)
            this_len += 1
            if i == "<sep>":
                break
        i = len(history) - 1
        while True:
            if history[i] == "<sep>":
                break
            i -= 1
        while True:
            i += 1
            if i == len(history):
                break
            print(history[i], end=' ', file=f1)
            this_len += 1
        print("", file=f1)
        maxlen = max(maxlen, this_len)


with open(f"{decoder}/test_data.txt", 'w') as f1:
    for history in test_set:
        this_len = 0
        for i in history:
            print(i, end=' ', file=f1)
            this_len += 1
            if i == "<sep>":
                break
        i = len(history) - 1
        while True:
            if history[i] == "<sep>":
                break
            i -= 1
        while True:
            i += 1
            if i == len(history):
                break
            print(history[i], end=' ', file=f1)
            this_len += 1
        print("", file=f1)
        maxlen = max(maxlen, this_len)

max_history = 0
with open(f"{chain}/train_data.txt", 'w') as f1:
    for history in train_set:
        max_history = max(max_history, len(history))
        for i in history:
            print(i, end=' ', file=f1)
        print("", file=f1)

with open(f"{chain}/test_data.txt", 'w') as f1:
    with open(f"{chain}/test_ans.txt", 'w') as f2:
        for history in test_set:
            for i in history:
                print(i, end=' ', file=f1)
                if i == "<sep>":
                    break
            print("", file=f1)
            i = len(history) - 1
            while True:
                if history[i] == "<sep>":
                    break
                i -= 1
            while True:
                i += 1
                if i == len(history):
                    break
                print(history[i], end=' ', file=f2)
            print("", file=f2)

print(f"max direct len:{maxlen}")
print(f"max cot len:{max_history}")
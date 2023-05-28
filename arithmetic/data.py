import random
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='data')

parser.add_argument('--file', type=str, default="Data")
parser.add_argument('--length', type=int, default=3)
parser.add_argument('--train_size', type=float, default=1e6)
parser.add_argument('--test_size', type=float, default=1e5)
parser.add_argument('--number_range', type=int, default=11)
parser.add_argument('--under', action='store_true', default=False)

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

def operator(a, b, rule):
    if rule == '+':
        x = (a + b) % rang
        return x
    if rule == '-':
        x = (a - b) % rang
        return x
    if rule == '*':
        x = mul[a, b]
        return x
    if rule == '/':
        x = div[a, b]
        return x

signs = ['+', '-', '*', '/']
random.seed(2023)

def get_expr(x):
    while True:
        num1 = random.choice(nums)
        sign = random.choice(signs)
        if sign == '+':
            num2 = operator(x, num1, '-')
        elif sign == '-':
            num2 = operator(num1, x, '-')
        elif sign == '*':
            if num1 == 0 and x != 0:
                continue
            elif num1 == 0:
                num2 = random.choice(nums)
            else:
                num2 = operator(x, num1, '/')
        else:
            if x == 0 and num1 == 0:
                num2 = random.choice(nums)
                if num2 == 0:
                    continue
            elif x == 0 or num1 == 0:
                continue
            else:
                num2 = operator(num1, x, '/')
        break
    return ['(', str(num1), sign, str(num2), ')']

def iter(lst):
    while True:
        item = random.randint(0, len(lst) - 1)
        if lst[item].isdigit():
            break
    expr = get_expr(int(lst[item]))
    need_bracket = False
    if item > 0 and lst[item - 1] == '/':
        need_bracket = True
    elif (item > 0 and lst[item - 1] == '*') or (item < (len(lst) - 1) and lst[item + 1] in ['*', '/']):
        if expr[2] in ['+', '-']:
            need_bracket = True
    elif item > 0 and lst[item - 1] == '-':
        if expr[2] in ['+', '-']:
            need_bracket = True
    del lst[item]
    if not need_bracket:
        expr = expr[1:-1]
    for i in reversed(expr):
        lst.insert(item, i)
    return lst

def get(length):
    lst = [str(random.choice(nums))]
    history = [lst[:]]
    for _ in range(length):
        lst = iter(lst)
        history.append(lst[:])
    ans = []
    for item in reversed(history):
        ans = ans + item + ["="]
    return ans[:-1]

maxlen = 4 * args.length + 1
maxhistory = 0
train_set = set()
if args.under:
    output_list = np.arange(args.length) + 1
    output_list = output_list / np.sum(output_list) * args.train_size
    for i in range(args.length):
        for _ in range(int(output_list[i])):
            history = get(i + 1)
            train_set.add(tuple(history))
            maxhistory = max(maxhistory, len(history))
while len(train_set) < args.train_size:
    history = get(args.length)
    train_set.add(tuple(history))
    maxhistory = max(maxhistory, len(history))

test_set = set()
while len(test_set) < args.test_size:
    history = get(args.length)
    tmp = tuple(history)
    if tmp not in train_set:
        test_set.add(tmp)
    maxhistory = max(maxhistory, len(history))

decoder = f"{args.file}/decoder"
chain = f"{args.file}/chain"
os.makedirs(f"{args.file}", exist_ok=True)
os.makedirs(decoder, exist_ok=True)
os.makedirs(chain, exist_ok=True)

with open(f"{decoder}/train_data.txt", 'w') as f1:
    for history in train_set:
        for i in history:
            print(i, end=' ', file=f1)
            if i == "=":
                break
        print(history[-1], file=f1)

with open(f"{decoder}/test_data.txt", 'w') as f1:
    for history in test_set:
        for i in history:
            print(i, end=' ', file=f1)
            if i == "=":
                break
        print(history[-1], file=f1)

with open(f"{chain}/train_data.txt", 'w') as f1:
    for history in train_set:
        for i in history:
            print(i, end=' ', file=f1)
        print("", file=f1)

with open(f"{chain}/test_data.txt", 'w') as f1:
    with open(f"{chain}/test_ans.txt", 'w') as f2:
        for history in test_set:
            for i in history:
                print(i, end=' ', file=f1)
                if i == "=":
                    break
            print("", file=f1)
            print(history[-1], file=f2)

print(f"max direct len:{maxlen}")
print(f"max cot len:{maxhistory}")

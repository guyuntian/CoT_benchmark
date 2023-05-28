import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader

class MyDataSet(Data.Dataset):
    def __init__(self, args, control):
        num_range = args.num_range
        variable = "abcde"
        dictionary = {"<pad>": 0, "=": 1, "<eos>": 2, "<sep>": 3, "+": 4, ",": 5}
        for i in range(len(variable)):
            dictionary[variable[i]] = i + 6
        for i in range(num_range):
            dictionary[str(i)] = i + 11
        debug_size = 100

        if not args.chain:
            if control == 0:
                with open(f"{args.file}/decoder/train_data.txt", 'r') as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
            elif control == 1:
                with open(f"{args.file}/decoder/test_data.txt", 'r') as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
        else:
            if control == 0:
                with open(f"{args.file}/chain/train_data.txt", 'r') as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
            elif control == 1:
                with open(f"{args.file}/chain/test_data.txt", 'r') as f:
                    self.X = f.read().splitlines()
                    if args.debug:
                        self.X = self.X[:debug_size]
                with open(f"{args.file}/chain/test_ans.txt", 'r') as f:
                    self.Y = f.read().splitlines()
                    if args.debug:
                        self.Y = self.Y[:debug_size]

        def toToken(sentences):
            token_list = list()
            for sentence in sentences:
                arr = [dictionary[s] for s in sentence.split()] + [2]
                padding = [0 for _ in range(args.maxdata - len(arr))]
                arr = arr + padding
                token_list.append(torch.Tensor(arr))
            return torch.stack(token_list).long()

        def toY(sentences):
            token_list = list()
            for sentence in sentences:
                arr = [dictionary[s] for s in sentence.split()]
                padding = [-1 for _ in range(args.maxans - len(arr))]
                arr = arr + padding
                token_list.append(torch.Tensor(arr))
            return torch.stack(token_list).long()

        def getY(X, chain):
            Y = X[:, 1:] * 1
            b = Y.shape[0]
            equa = torch.argmax(torch.where(Y==dictionary['<sep>'], 1, 0), dim=1)
            eos = torch.argmax(torch.where(Y==dictionary['<eos>'], 1, 0), dim=1)
            for i in range(b):
                Y[i, :equa[i] + 1] = 0
                Y[i, eos[i]+1:] = 0
            return Y

        self.X = toToken(self.X)
        self.Y = toY(self.Y) if args.chain and (control != 0) else getY(self.X, args.chain)
        if not (args.chain and (control != 0)):
            self.X = self.X[:,:-1]
        self.Z = torch.argmax(torch.where(self.X==dictionary['<sep>'], 1, 0), dim=1)
  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]  

def getLoader(args):
    number = 2
    datasets = [MyDataSet(args, i) for i in range(number)]
    samplers = [torch.utils.data.distributed.DistributedSampler(datasets[i]) for i in range(number)]
    dataloaders = [DataLoader(datasets[i], batch_size=args.batch_size, shuffle=False,\
                num_workers=10, drop_last=False, sampler=samplers[i], pin_memory=True) for i in range(number)]
    if not args.chain:
        return dataloaders[0], dataloaders[1]
    else:
        return dataloaders[0], dataloaders[1]

import torch
import torch.distributed as dist

def evaluate(model, cur_loader):
    Sum, correct = 0, 0
    cur_loader.sampler.set_epoch(0)
    for input_ids, y, z in cur_loader:
        inputs, y, z = input_ids.long().cuda(), y.cuda(), z.cuda()
        idx = model.generate(inputs, z) 
        eos = torch.argmax(torch.where(idx == 2, 1, 0), dim=1)
        expected_length = torch.sum(torch.where(y != -1, 1, 0), dim=1)
        start = eos - expected_length - 1
        for i in range(input_ids.shape[0]):
            if idx[i, start[i]] != 3:
                continue
            if torch.all(y[i, :expected_length[i]] == idx[i, start[i]+1:eos[i]]):
                correct += 1 
        Sum += inputs.shape[0]
    correct = torch.as_tensor(correct).cuda()
    Sum = torch.as_tensor(Sum).cuda()
    dist.all_reduce(correct)
    dist.all_reduce(Sum)
    return correct / Sum
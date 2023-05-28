import torch
import torch.distributed as dist

def evaluate(model, cur_loader):
    Sum, correct = 0, 0
    cur_loader.sampler.set_epoch(0)
    for input_ids, y, _ in cur_loader:
        inputs, y = input_ids.cuda(), y.cuda()
        logits = model(inputs)
        Sum += torch.as_tensor(inputs.shape[0]).cuda()
        truth = torch.where(y > 0, 1, 0)
        predict = torch.where(torch.argmax(logits, dim=2) == y, 1, 0) * truth
        correct += torch.sum(torch.where(torch.sum(truth, dim=1) == torch.sum(predict, dim=1), 1, 0))
    dist.all_reduce(correct)
    dist.all_reduce(Sum)
    return correct / Sum
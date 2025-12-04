
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from seed_utils import set_seed, DEVICE
from data_cifar10 import get_cifar10_loaders
from model_mobilenetv2 import mobilenet_v2_cifar10

class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps
    def forward(self, preds, target):
        logp = F.log_softmax(preds, dim=1)
        return -(logp.gather(1, target.unsqueeze(1)).squeeze(1)*(1-self.eps)
                + logp.mean(dim=1)*self.eps).mean()

def cosine_with_warmup(opt, base_lr, warmup_epochs, total_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep+1)/float(warmup_epochs)
        prog = (ep - warmup_epochs)/max(1, total_epochs - warmup_epochs)
        return 0.5*(1 + math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def evaluate(model, loader):
    crit = nn.CrossEntropyLoss()
    model.eval()
    loss_sum=0; correct=0; n=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out, y)
            loss_sum += loss.item()*y.size(0)
            correct += (out.argmax(1)==y).sum().item()
            n += y.size(0)
    return loss_sum/n, 100.0*correct/n

def train_baseline(epochs=200, lr=0.2, wd=5e-4):
    set_seed(42)
    trainloader, valloader, testloader = get_cifar10_loaders()

    model = mobilenet_v2_cifar10()
    model.to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=wd, nesterov=True)
    sched = cosine_with_warmup(opt, lr, warmup_epochs=5, total_epochs=epochs)
    crit = LabelSmoothingCE(eps=0.1)

    hist=[]
    for ep in range(epochs):
        model.train()
        for x,y in trainloader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
        sched.step()

        tr_loss, tr_acc = evaluate(model, trainloader)
        va_loss, va_acc = evaluate(model, valloader)
        hist.append((ep+1, tr_acc, va_acc))

        print(f"Epoch {ep+1}/{epochs}  val_acc={va_acc:.2f}%")

    torch.save({"state_dict":model.state_dict()}, "checkpoints/baseline_fp32.pt")

    return model, hist

import torch
from tqdm import tqdm

# 精度计算函数
def cal_accuracy(model, loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = correct/total
    # print("target:", target[:5])
    # print("output:", output[:5])
    # print("predicted:", predicted[:5])
    return accuracy

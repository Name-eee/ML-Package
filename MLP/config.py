import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# 绘图
def draw_logs(epoch_log, loss_log=[], acc_log=[], title=''):
    fig, ax1 = plt.subplots()
    ax1.plot(epoch_log, loss_log, 'r-', label='loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(epoch_log, acc_log, 'b-', label='acc')
    ax2.set_ylabel('acc')
    ax2.tick_params(axis='y')

    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    
    plt.legend(handles=handles, labels=labels, loc='upper left')
    fig.tight_layout()
    plt.show()

# 计算精度
def cal_accuracy(y_pred, y_label):
    y_pred_res=np.argmax(y_pred,axis=1)
    correct = (y_pred_res == y_label).sum().item()
    total = len(y_pred_res)
    accuracy = correct/total
    # print(correct)
    return accuracy

# 分离数据集
def train_test_split(dataset, test_p=1/3):
    sum_len = len(dataset.data)
    train_len = math.ceil(sum_len*(1-test_p))
    train_idx = np.random.choice(np.arange(sum_len), replace=False, size=train_len)
    train_data = {'data': dataset.data[train_idx], 'target':dataset.target[train_idx]}
    
    test_idx = np.array([i for i in range(sum_len) if i not in train_idx])
    test_data = {'data': dataset.data[test_idx], 'target':dataset.target[test_idx]}
    print(f"All:{sum_len}, trainset:{len(train_data['data'])}, testset:{len(test_data['data'])}.")
    return train_data, test_data

# 加载数据集
def dataloader(dataset, batch_size=32):
    data = []
    target=[]
    sum_len=len(dataset['data'])
    batch=sum_len//batch_size
    idx_last=list(range(sum_len))
    while len(idx_last) > batch_size:
        idx_selected=np.random.choice(idx_last, batch_size, replace=True)
        data.append(dataset['data'][idx_selected])
        target.append(dataset['target'][idx_selected])
        idx_last = [i for i in idx_last if i not in idx_selected]
    data.append(dataset['data'][idx_last])
    target.append(dataset['target'][idx_last])
    return data, target

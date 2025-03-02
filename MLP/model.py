# import
import numpy as np

# 多层感知机（forward中可以自定义层数）
class MLP:
    def __init__(self, lr=0.01, mode='train', feature_num=4, variety=3, initr=0.01):
        self.lr=lr
        self.mode=mode
        if self.mode=='train':
            self.weights_last=[]
            self.bias_last=[]
        self.fc1 = self.linear(feature_num, 64)
        self.fc2 = self.linear(64, 32)
        self.fc3 = self.linear(32, variety)
        self.sft = self.softmax()
        self.relu = self.ReLU()
        self.initr = initr

    def linear(self, in_size, out_size):
        def linear_module(x):
            w = np.random.uniform(low=-self.initr, high=self.initr, size=(out_size, in_size))
            b = np.random.uniform(low=-self.initr, high=self.initr, size=(out_size))
            if self.mode=='train' and self.weights_last==[]:
                self.xs.append(np.array(x))
                self.weights.append(np.array(w))
                self.bias.append(np.array(b))
                self.grads.append(np.array(w))
            elif self.mode=='train':
                w=self.weights_last.pop()
                b=self.bias_last.pop()
                self.xs.append(np.array(x))
                self.weights.append(np.array(w))
                self.bias.append(np.array(b))
                self.grads.append(np.array(w))
            else:
                w=self.weights.pop()
                b=self.bias.pop()
            # print(x.shape)
            # print("w:", w.shape)
            return np.dot(x, w.T)+b
        return linear_module
    
    def softmax(self):
        def func(x):
            s=np.exp(x)/np.sum(np.exp(x), axis=1).reshape(1,-1).T
            return s
        return func
    
    def ReLU(self):
        def func(x):
            if self.mode=='train':
                self.ReLU_grad=(x>0).astype(int)
            return np.max(0,x)
        return func

    def CEloss(self, y_pred, y_label):
        N = len(y_label)
        # print("N:", N)
        y_true = np.zeros_like(y_pred)
        for i in range(y_true.shape[0]):
            y_true[i][y_label[i]]=1
        return -1/N*np.sum(y_true*np.log(y_pred+0.001))

    def cal_SCgrad(self, s, y_label):
        scgrad=s.copy()
        for i in range(len(y_label)):
            scgrad[i][y_label[i]]=scgrad[i][y_label[i]]-1
        return np.array(scgrad)*self.lr

    def forward(self, x):
        x = self.fc1(x)
        # print("1", x.shape)
        x = self.fc2(x)
        # print("2", x.shape)
        x = self.fc3(x)
        # print("3", x.shape)
        # print("3.5", x[:3])
        x = self.sft(x)
        # print("4", x[:3])
        return x
    
    def backward(self):
        # grads
        # print("SC_grads:",  self.grads[-1][0])
        # print("SC_grads:",  self.grads[-1].shape)
        # print("weights:", [i.shape for i in self.weights])
        # print("xs:", [i.shape for i in self.xs])
        # print("grads:", [i.shape for i in self.grads])
        # print("bias:", [i.shape for i in self.bias])
        # print("grads_max:",  np.max([np.max(i) for i in self.grads]))
        # print("grads_min:", np.min([np.min(i) for i in self.grads]))
        grads_len = len(self.grads)
        for i in range(grads_len-2,0,-1):
            self.grads[i]=np.dot(self.grads[i+1], self.grads[i])
        self.grads.pop(0)
        # print()
        # print("grads:", [i.shape for i in self.grads])
        # print("bias:", [i.shape for i in self.bias])
        # print("grads_max:",  np.max([np.max(i) for i in self.grads]))
        # print("grads_min:", np.min([np.min(i) for i in self.grads]))

        # weights
        self.weights=[
                    self.weights[i]-self.lr*np.dot(self.grads[i].T, self.xs[i])
                    for i in np.arange(len(self.weights))
                    ]
        self.weights_last=self.weights.copy()
        self.weights_last.reverse()

        # bias
        self.bias=[
                    self.bias[i]-self.lr*np.dot(self.grads[i].T, np.ones([self.grads[i].shape[0], 1])).reshape(-1)
                    for i in np.arange(len(self.bias))
                    ]
        self.bias_last=self.bias.copy()
        self.bias_last.reverse()

    def train(self, x, y):
        self.mode='train'
        self.weights=[]
        self.bias=[]
        self.xs=[]
        self.grads=[]

        x = self.forward(x)
        loss = self.CEloss(x, y)
        self.SC_grad=self.cal_SCgrad(x, y)
        self.grads.append(self.SC_grad)
        self.backward()
        # print("y_pred:", x[0])
        # print("y_label:", y[0])
        self.grad_log = self.grads

        self.weights=[]
        self.bias=[]
        self.xs=[]
        self.grads=[]
        return x, loss
    
    def test(self, x, y):
        self.mode='test'
        self.weights=self.weights_last.copy()
        self.bias=self.bias_last.copy()
        x = self.forward(x)
        loss = self.CEloss(x, y)
        # print("weights:", [i.shape for i in self.weights])
        # print("weights_last:", [i.shape for i in self.weights_last])
        return x, loss

    def save(self, path):
        return path
    
    def load(self, path):
        return path
    
    def clear(self):
        self.weights_last=[]
        self.bias_last=[]
        self.weights=[]
        self.bias=[]
        self.xs=[]
        self.grads=[]
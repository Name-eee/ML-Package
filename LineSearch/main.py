import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sympy import symbols, sin, cos
from scipy.optimize import minimize
from tqdm import tqdm
import time

xrange=[-2, 2]

def obj_func(x):
    x0 = np.array(x)
    x = x0[0]
    y = x0[1]
    def penalty(x1, k=1):
        if x1 < -2:
            return k * (-2 - x1)**4
        elif x1 > 2:
            return k * (x1 - 2)**4
        else:
            return 0
    return np.sin(x)*np.cos(y) + 0.5 * np.cos(2*x) * np.sin(2*y) + penalty(x) + penalty(y)

def draw_3D(func, point=[]):
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    x = np.c_[X1.ravel(), X2.ravel()]
    y = np.array([func(i) for i in x])
    y = y.reshape(len(x1), len(x2))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, y, cmap='viridis')
    if point != []:
        ax.scatter(point[0], point[1], point[2], c='y')
        ax.scatter(point[0][0], point[1][0], point[2][0], c='r')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Z')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def draw_contour(func, point=np.array([])):
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    x = np.c_[X1.ravel(), X2.ravel()]
    y = np.array([func(i) for i in x])
    y = y.reshape(len(x1), len(x2))
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, y, levels=20, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.colorbar(label="Function Value")
    # 绘制路径
    plt.scatter(point[0][0], point[0][1], c='yellow', label='start')
    plt.scatter(point[-1][0], point[-1][1], c='orange', label='end')
    plt.plot(point[:, 0], point[:, 1], c='r', label='log')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

def grad(f, x, y, h=1e-5):
    df_dx = (f([x + h, y]) - f([x, y])) / h  # 对 x 求偏导
    df_dy = (f([x, y + h]) - f([x, y])) / h  # 对 y 求偏导
    return np.array([df_dx, df_dy])

def hessian(f, x, y, h=1e-5):
    d2f_dx2 = (f([x + h, y]) - 2*f([x, y]) + f([x - h, y])) / h**2
    d2f_dy2 = (f([x, y + h]) - 2*f([x, y]) + f([x, y - h])) / h**2
    d2f_dxdy = (f([x + h, y + h]) - f([x + h, y]) - f([x, y + h]) + f([x, y])) / (h**2)
    hessian_matrix = np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])
    return hessian_matrix

# 搜索方向p
def steepest(func, xk):
    return -grad(func, xk[0], xk[1])

def newton(func, xk, gamma=1):
    I = np.eye(len(xk))
    func_grad = grad(func, xk[0], xk[1])
    func_hessian = hessian(func, xk[0], xk[1])+gamma*I
    return -np.linalg.inv(func_hessian).dot(func_grad)

def BFGS(func, xk_1, xk_0, Bk_0, gamma=1e-6):
    # print(xk_1-xk_0)
    func_grad = grad(func, xk_1[0], xk_1[1])
    if np.sum((xk_1-xk_0)**2) < 1e-4:
        Bk = hessian(func, xk_1[0], xk_1[1]) + gamma*np.eye(len(xk_0))
        p = -np.linalg.inv(Bk).dot(func_grad)
    else:
        func_grad0 = grad(func, xk_0[0], xk_0[1])
        sk = xk_1 - xk_0
        yk = func_grad - func_grad0
        b1 = np.dot(Bk_0, sk)
        b2 = np.dot(sk.T, Bk_0)
        Bk = Bk_0 - np.dot(b1, b2) / np.dot(sk.T, yk) + np.dot(yk, yk) / np.dot(yk.T, sk)
        p = -np.linalg.inv(Bk).dot(func_grad)
    return p, Bk


# 搜索步长a
def armijo(func, xk, pk, alpha=1, c1=1e-4):
    func_now = func(xk + alpha * pk)
    cond_now = func(xk) + c1 * alpha * np.dot(grad(func, xk[0], xk[1]), pk)
    # print("xk ", xk)
    # print("pk ", pk)
    # print("xk+apk", xk + alpha * pk)
    # print("func", func_now)
    # print("cond", cond_now)
    # print()
    if func_now > cond_now:
        return 1
    else:
        return 0

def armijo_body(func, xk, pk, alpha=1, c1=1e-4, beta=0.1):
    a = armijo(func, xk, pk, alpha, c1)
    epoch = 1000
    while a:
        alpha*=beta
        a = armijo(func, xk, pk, alpha, c1)
        epoch-=1
        if epoch <= 0:
            print('Epoch stop.')
            break
    return alpha

def curvature(func, xk, pk, alpha=1, c2=0.9):
    newx = xk+alpha*pk
    func_now = np.dot(grad(func, newx[0], newx[1]).T, pk)
    cond_now = c2 * np.dot(grad(func, xk[0], xk[1]), pk)
    if func_now < cond_now:
        return 1
    else:
        return 0

def armijo_body(func, xk, pk, alpha=0.01, c1=1e-4, beta=0.01):
    c = curvature(func, xk, pk, alpha, c)
    epoch = 1000
    while c:
        alpha+=beta
        c = curvature(func, xk, pk, alpha, c)
        epoch-=1
        if epoch <= 0:
            print('Epoch stop.')
            break
    return alpha

def fai_alpha(func, xk, pk, c1=1e-4, arange=[0,1]):
    func_log = []
    cond_log = []
    armijo_log = []
    curvature_log = []
    alphas = np.linspace(arange[0], arange[1], 1000)
    for alpha in alphas:
        func_now = func(xk + alpha * pk)
        cond_now = func(xk) + c1 * alpha * np.dot(grad(func, xk[0], xk[1]), pk)
        func_log.append(func_now)
        cond_log.append(cond_now)
        armijo_log.append(armijo(func, xk, pk, alpha))
        curvature_log.append(curvature(func, xk, pk, alpha))
    return alphas, func_log, cond_log, armijo_log, curvature_log

def draw_fai(alphas, func_log, cond_log, a_log, c_log):
    plt.plot(alphas, func_log, 'r-')
    plt.plot(alphas, cond_log, 'b-')
    for idx in range(0, len(a_log)-1):
        # if a_log[idx] != a_log[idx+1]:
        if a_log[idx] == 0:
            plt.axvline(x=alphas[idx], color=(1, 1, 0 , 0.1), linestyle='-')
    plt.axvline(x=alphas[idx], color=(1, 1, 0 , 0.5), linestyle='-', label='armijo')

    for idx in range(0, len(c_log)-1):
        # if c_log[idx] != c_log[idx+1]:
        if c_log[idx] == 0:
            plt.axvline(x=alphas[idx], ymin=0.2, ymax=0.8, color=(1, 0.8, 0 , 0.1), linestyle='-')
    plt.axvline(x=alphas[idx], color=(1, 0.8, 0 , 0.5), linestyle='-', label='curvature')
    plt.legend()
    plt.show()

def wolfe(func, xk, pk, alpha=2, c1=1e-3, c2=0.9, beta1=0.8, beta2=0.01):
    a = armijo(func, xk, pk, alpha, c1)
    c = curvature(func, xk, pk, alpha, c2)
    epoch = 1000
    alpha_log = []
    while a or c:
        if a == 1:
            alpha*=beta1
        alpha+=beta2*c
        # print(a, c, alpha)
        a = armijo(func, xk, pk, alpha)
        c = curvature(func, xk, pk, alpha)
        alpha_log.append([a, c, alpha])
        epoch-=1
        if epoch <= 0 or alpha<1e-5:
            # print('Wolfe stop.')
            # for i in alpha_log:
            #     print(i)
            alog, clog = [], []
            for alpha in np.array(alpha_log)[:, 2]:
                alog.append(armijo(func, xk, pk, alpha))
                clog.append(curvature(func, xk, pk, alpha))
            alphas, func_log, cond_log, alog, clog = fai_alpha(func, xk, pk, arange=[0, 2])
            # draw_fai(alphas, func_log, cond_log, alog, clog)
            alpha = alpha + 0.05
            break
    return alpha

def line_search(func, pk_func=BFGS, alpha_func=wolfe, xk=[0.1, 0.1], epochs=30, alpha=1, c1=1e-4, c2=0.9, beta1=0.1, beta2=0.01, log=False):
    xk = np.array(xk)
    Bk = hessian(func, xk[0], xk[1])
    pk = newton(func, xk)
    alpha = alpha_func(func, xk, pk)
    if log == True:
        x_log = []
        y_log = []
    sum_time = 0
    for epoch in range(epochs):
        xk_0 = xk.copy()
        xk = xk + alpha*pk
        start = time.time()
        if pk_func == BFGS:
            pk, Bk = BFGS(func, xk, xk_0, Bk)
        else:
            pk = pk_func(func, xk)
        end = time.time()
        sum_time += (end-start)
        alpha = alpha_func(func, xk, pk)
        # print(alpha)
        y = func(xk)
        # print(f"Epoch {epoch}, x:({xk[0]:.4f},{xk[1]:.4f}), y:{y}")
        if log == True:
            x_log.append(xk)
            y_log.append(y)
    if log == False:
        return func(xk), xk, alpha, pk, sum_time
    else:
        return func(xk), xk, alpha, pk, x_log, y_log, sum_time

if __name__ == '__main__':
    # # draw_3D(obj_func)
    # # draw_contour(obj_func)
    # xk0 =[0.1, 0.1]
    # pk0 = steepest(obj_func, xk0)
    # y0 = obj_func(xk0)
    # alphas, func_log, cond_log, alog, clog = fai_alpha(obj_func, xk0, pk0, arange=[0, 2])
    # draw_fai(alphas, func_log, cond_log, alog, clog)
    # 计算
    xks=[[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1]]
    yend=1e5
    sum_time = 0
    for xk0 in xks:
        # start_time = time.time()
        # result = minimize(obj_func, xk0, method='BFGS')
        # end_time = time.time()
        # xk = result.x
        # y = result.fun
        # times = end_time - start_time
        y, xk, alpha, pk, x_log, y_log, times = line_search(obj_func, pk_func=BFGS, xk=xk0, epochs=50, alpha=1, log=True)
        y, xk, alpha, pk, times = line_search(obj_func, pk_func=BFGS, xk=xk0, epochs=20, alpha=1)
        sum_time+=times*1000
        print(f"x:({xk[0]:.4f},{xk[1]:.4f}), y:{y}")
        print(f"Execution_time:{times*1000:.4f}ms")
        if y < yend:
            yend = y
            xkend = xk
        # draw_3D(obj_func, point=[[xkend[0], xk0[0]], [xkend[1], xk0[1]], [yend, y0]])
        # print(x_log)        
        # draw_contour(obj_func, point=np.array(x_log))
        print()
    print(f"x:{xkend}, y:{yend}")
    print(f"Avg_time:{sum_time/4:.4f}ms")
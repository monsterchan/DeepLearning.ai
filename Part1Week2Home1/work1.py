import math
def basic_sigoid(x):
    return 1/(1+math.exp(-x))
print(basic_sigoid(3) )

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

x = np.array([1,2,3])
print(sigmoid(x))
# [0.26894142 0.11920292 0.04742587]

#练习：创建函数sigmoid_grad（）计算sigmoid函数相对于其输入x的梯度。 公式为：
# sigmoid\_derivative(x) = \sigma'(x) = \sigma(x) (1 - \sigma(x))\tag{2}
# 我们通常分两步编写此函数代码：
# 1.将s设为x的sigmoid。 你可能会发现sigmoid（x）函数很方便。
# 2.计算\sigma'(x) = s(1-s)
def sigmoid_derivative(x):
    # 计算S型函数相对于其输入x的梯度（也称为斜率或导数）。 您可以将S型函数的输出存储到变量中，然后使用它来计算梯度。
    s = sigmoid(x)
    ds = s *(1-s)
    # ds - 计算出的梯度
    return  ds

x =np.array([1,2,3])
print("sigmoid_derivative="+ str(sigmoid_derivative(x)))
# sigmoid_derivative=[0.19661193 0.10499359 0.04517666]

# 1.3- 重塑数组
# 深度学习中两个常用的numpy函数是np.shape和np.reshape()。
# -X.shape用于获取矩阵/向量X的shape（维度）。
# -X.reshape（...）用于将X重塑为其他尺寸。

def image2vector(image):
    # image - 形状（长度，高度，深度）的数字数组
    # v - 形状的向量（长 * 高 * 深，1）
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image2vector(image) = " + str(image2vector(image)))

# image2vector(image) = [[0.67826139]
#  [0.29380381]
#  [0.90714982]
#  [0.52835647]
#  [0.4215251 ]
#  [0.45017551]
#  [0.92814219]
#  [0.96677647]
#  [0.85304703]
#  [0.52351845]
#  [0.19981397]
#  [0.27417313]
#  [0.60659855]
#  [0.00533165]
#  [0.10820313]
#  [0.49978937]
#  [0.34144279]
#  [0.94630077]]

# 1.4- 行标准化
# 我们在机器学习和深度学习中使用的另一种常见技术是对数据进行标准化。 由于归一化后梯度下降的收敛速度更快，通常会表现出更好的效果。

def normalizeRows(x):
    # 实现一个对矩阵x的每一行进行规范化（具有单位长度）的函数。
    # 输入： x - 形状为（n，m）的numpy矩阵
    # 输出：x - 规范化（按行）的numpy矩阵。
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)

    x=x/x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))

# normalizeRows(x) = [[0.         0.6        0.8       ]
#                     [0.13736056 0.82416338 0.54944226]]


# 1.5- 广播和softmax函数
def softmax(x):
    # 计算输入x的每一行的softmax。 用于行向量以及形状（n，m）的矩阵
    # x - 形状为（n，m）的numpy矩阵
    # 返回值： s - numpy矩阵，等于形状（n，m）的x的softmax

    # 将exp（）逐个元素应用于x。使用np.exp（...）。
    x_exp = np.exp(x)

    # 创建一个向量x_sum，该向量求和x_exp的每一行。使用np.sum（...，axis = 1，keepdims = True）。
    x_sum = np.sum(x_exp,axis=1,keepdims=True)

    # 通过将x_exp除以x_sum来计算softmax（x）。它应该自动使用numpy广播。
    s = x_exp / x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))

# softmax(x) = [[9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04
#   1.21052389e-04]
#  [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04
#   8.01252314e-04]]

import time
x1 = np.random.rand(1)
x2 = np.random.rand(1)



# 矢量实施的经典点积
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i]*x2[i]
toc = time.process_time()
print("dot="+str(dot)+"\n------dot所用时间="+str((toc-tic)*1000)+"ms")

tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- dot向量化后所耗时间 = " + str(1000*(toc - tic)) + "ms")

x1 = np.random.rand(0)
x2 = np.random.rand(0)

tic = time.process_time()
#创建大小为(len(x1),len(x2)) 的零矩阵
size = len(x1)
outer = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print("outer="+str(outer)+"\n----outer所耗时间="+str(1000*(toc-tic))+"ms")

tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print("outer="+str(outer)+"\n-----outer向量化后所耗时间=" + str(1000*(toc - tic)) + "ms" )

x1 = np.random.rand(0)
x2 = np.random.rand(0)

tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- elementwise所耗时间 = " + str(1000*(toc - tic)) + "ms")

tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print("mul="+str(mul)+"\n-----elementwise向量化后所耗时间=" + str(1000*(toc - tic)) + "ms" )


W = np.random.rand(3,len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- gdot所耗时间 = " + str(1000*(toc - tic)) + "ms")


tic = time.process_time()
dot = np.multiply(W,x1)
toc = time.process_time()
print("gdot="+str(dot)+"\n-----gdot向量化后所耗时间=" + str(1000*(toc - tic)) + "ms" )
# dot=250084.76006018423
# ------dot所用时间=750.0ms
# dot = 250084.76006018213
#  ----- dot向量化后所耗时间 = 31.25ms
# outer=[[0.02071126 0.09555269 0.14442706 ... 0.11184553 0.08170454 0.13806498]
#  [0.08205103 0.37854758 0.57217136 ... 0.44309433 0.32368589 0.54696697]
#  [0.05765745 0.26600627 0.40206616 ... 0.31136343 0.22745483 0.38435497]
#  ...
#  [0.0781874  0.36072247 0.54522887 ... 0.42222984 0.30844412 0.5212113 ]
#  [0.09412523 0.43425265 0.6563691  ... 0.50829777 0.3713178  0.62745576]
#  [0.08109653 0.37414392 0.56551528 ... 0.43793981 0.31992044 0.54060409]]
# ----outer所耗时间=73968.75ms
# outer=[[0.02071126 0.09555269 0.14442706 ... 0.11184553 0.08170454 0.13806498]
#  [0.08205103 0.37854758 0.57217136 ... 0.44309433 0.32368589 0.54696697]
#  [0.05765745 0.26600627 0.40206616 ... 0.31136343 0.22745483 0.38435497]
#  ...
#  [0.0781874  0.36072247 0.54522887 ... 0.42222984 0.30844412 0.5212113 ]
#  [0.09412523 0.43425265 0.6563691  ... 0.50829777 0.3713178  0.62745576]
#  [0.08109653 0.37414392 0.56551528 ... 0.43793981 0.31992044 0.54060409]]
# -----outer向量化后所耗时间=484.375ms
# elementwise multiplication = [0.03193241 0.06713531 0.5331545  ... 0.00178877 0.68138161 0.07755435]
#  ----- elementwise所耗时间 = 796.875ms
# mul=[0.03193241 0.06713531 0.5331545  ... 0.00178877 0.68138161 0.07755435]
# -----elementwise向量化后所耗时间=0.0ms
# gdot = [250033.57823197 249767.3662673  249430.81196528]
#  ----- gdot所耗时间 = 3093.75ms
# gdot=[[3.44156976e-02 1.74716384e-01 8.92561385e-02 ... 3.02648090e-04
#   2.17920834e-01 3.53817508e-01]
#  [1.84960864e-02 1.78308842e-01 5.91351208e-01 ... 2.92963298e-03
#   5.09580532e-01 3.10458708e-01]
#  [1.89789143e-02 3.41281921e-01 2.54714634e-01 ... 1.31974216e-02
#   4.39869158e-01 1.22641198e-01]]
# -----gdot向量化后所耗时间=15.625ms


def L1(yhat,y):
    # yhat - 大小为m的向量（预测的标签）
    # y - 大小为m的向量（真标签）
    # 返回值： loss - 上面定义的L1损失函数的值
    loss = np.sum(np.abs(y - yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

# L1 = 1.1

def L2(yhat, y):
    """
    yhat-大小为m的向量（预测的标签）
    y-大小为m的向量（真标签）
    返回值： loss-上面定义的L2损失函数的值
    """
    loss = np.dot((y - yhat), (y - yhat).T)            #  .T表示转置
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))

# L2 = 0.43
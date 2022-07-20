import numpy as np
context1 = np.load('/public/others/lengyan/ly-4/kd1000.npy')
context1 = context1.reshape(-1,128)
context2 = np.load('/public/others/lengyan/ly-4/logmel.npy')
print(context2.shape)
# print (context1.shape)
# print (context2.shape)
c = np.zeros((2880,40))
# c = []
# print (c.shape)
for i in range(2880):
    for j in range(469):
        for m in range(40):
            a = context1[m,:]
            b = context2[i,j,:]
            distance = np.sqrt(np.sum(np.square(a - b)))
            if distance == 0:
                c[i,m] = c[i,m] + 1
print (c.shape)#2880,40
# np.save('/public/others/lengyan/ly-3/GX/tram3.npy',c)
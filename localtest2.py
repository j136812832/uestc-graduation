import numpy as np
import matplotlib.pyplot as plt
import time

# a = np.arange(0.0, 5.0, 0.02)
# print(type(a))
listx = [1.0, 2.0, 3, 4, 5]
listy = [1, 2, 3, 4, 5]
def plotpic():
    global listx
    global listy
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(listx, listy)
    plt.savefig("./pic/test.jpg")
    plt.show()

# plotpic()

now = time.time()
plotpic()
last = time.time()
middle = last - now
print(middle)
print("middle is {}".format(middle))
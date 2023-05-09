import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei' # 黑体

'''acc = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,
       47,48,49,50]
Y1 = [54.38,8.16,65.06,76.45,81.38,84.04,85.02,55.01,91,92,73,76,38,26,31]'''
app = [1,5,10,15,20,25,30,35,40,45,50]
Y2 = [28.76,86.47,90.42,93.24,94.09,94.76,95.39,95.34,95.94,96.09,96.63]



plt.figure()

# X1的分布
'''plt.plot(acc, Y1, label="无恶意用户", color="#3399FF", marker='*', linestyle="-")'''

# X2的分布
plt.plot(app, Y2,label="10%的恶意用户",  color="#FF3B1D", marker='o', linestyle="-")

# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))   # 将横坐标的值全部显示出来
X_labels = ['1','5','10','15','20','25','30','35','40','45','50']
plt.xticks(app,X_labels,rotation=0)
plt.legend()
plt.title("")
plt.xlabel("迭代次数")
plt.ylabel("模型准确率")
plt.show()


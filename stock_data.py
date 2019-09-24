import pandas as pd
import numpy as np
import math
import time
from itertools import chain
t1 = time.time()
load_path = r'C:\Users\SRJRZD\Desktop\平安银行20170601.csv'
save_path = r'C:\Users\SRJRZD\Desktop\处理后平安银行20170601.csv'
data = pd.read_csv(load_path, encoding='gbk')
Time = []
point = 0
for t in data.时间:
    if len(t) == 7:
        temp = int(t[0:1])*3600+int(t[2:4])*60+int(t[5:7])
    else:
        temp = int(t[0:2])*3600+int(t[3:5])*60+int(t[6:8])
    Time.append(temp)
# 寻找连续竞价开始点
for i in range(0, len(Time)):
    if Time[i] >= 34200:
        point = i
        break
Time = pd.Series(Time)
lcp = 9.20  # 昨日收盘价


# 股票价格与该价格在数列中位置的对应关系
def tran(obj, signal):
    if signal == 0:
        return int(round(obj*100-math.ceil(lcp*90)))  # python浮点数计算有点麻烦。。
    if signal == 1:
        return round((obj+math.ceil(lcp*90))/100, 2)


volume1 = np.zeros(2*int(lcp*10)+1)
volume2 = np.zeros(2*int(lcp*10)+1)
order = np.zeros(2*int(lcp*10)+1)  # 记录了相应价格的买卖数量，买为正，卖为负
length0 = len(volume1)  # 可能出现的价格数量
# 汇总集合竞价的买卖单
for i in range(0, point):
    if data.买卖方向[i] == '2B':
        volume1[tran(data.价格[i], 0)] += data.数量[i]
    if data.买卖方向[i] == 'BC':
        volume1[tran(data.价格[i], 0)] -= data.数量[i]
    if data.买卖方向[i] == '2S':
        volume2[tran(data.价格[i], 0)] += data.数量[i]
    if data.买卖方向[i] == 'SC':
        volume2[tran(data.价格[i], 0)] -= data.数量[i]
open_B = volume1.copy()  # 集合竞价的买单
open_S = volume2.copy()  # 集合竞价的卖单
for i in range(0, length0):
    volume1[i] = sum(volume1[i:length0])
    volume2[length0-i-1] = sum(volume2[0:length0-i])
open_v = 0  # 开盘价
index = 0
# 寻找最大成交量时的价格
for i in range(0, length0):
    if volume1[i] > volume2[i]:
        temp = volume2[i]
    else:
        temp = volume1[i]
    if temp > open_v:
        open_v = temp
        index = i
# 寻找可能的开盘价
for i in range(index+1, length0):
    if volume1[i] > volume2[i]:
        temp = volume2[i]
    else:
        temp = volume1[i]
    if temp < open_v:
        end = i-1
        break
temp = float('inf')
# 最接近昨日收盘价的即为今日开盘价
for i in range(index, end+1):
    if abs(tran(i, 1)-lcp) < temp:
        temp = abs(tran(i, 1)-lcp)
        index_temp = i
open_v = round(open_v, 2)
open_p = tran(index_temp, 1)  # 开盘成交量
# 生成集合竞价后的订单簿
for i in range(0, index):
    order[i] = open_B[i]
for i in range(index+1, length0):
    order[i] = -open_S[i]
for i in range(index, -1, -1):
    if volume1[i] > volume1[index]:
        index_B = i+1
        break
for i in range(index, length0):
    if volume2[i] > volume2[index]:
        index_S = i-1
        break
if index_B == index_S:
    order[index] = volume1[index]-volume2[index]
else:
    order[index_B] = volume1[index_B]-open_v
    order[index_S] = -(volume2[index_S]-open_v)
data = data[point:]  # 删去集合竞价阶段数据
data = data.reset_index(drop=True)
num = len(data.时间)  # 总订单数
buy10_p = np.full(10, np.nan)  # 买10价
buy10_v = np.full(10, np.nan)  # 买10量
sell10_p = np.full(10, np.nan)  # 卖10价
sell10_v = np.full(10, np.nan)  # 卖10量
buy_spread = np.full(10, np.nan)  # 买价价差
sell_spread = np.full(10, np.nan)  # 卖价价差
mean10 = np.full(4, np.nan)
total_diff = np.full(2, np.nan)
for i in range(0, 10):
    data['买'+str(i+1)+'价'] = np.zeros(num)
    data['买'+str(i+1)+'量'] = np.zeros(num)
for i in range(0, 10):
    data['卖'+str(i+1)+'价'] = np.zeros(num)
    data['卖'+str(i+1)+'量'] = np.zeros(num)
for i in range(0, 10):
    data['价差'+str(i+1)] = np.zeros(num)
for i in range(0, 10):
    data['中间价'+str(i+1)] = np.zeros(num)
data['总买10价差'] = np.zeros(num)
for i in range(0, 9):
    data['买10价差'+str(i+1)] = np.zeros(num)
data['总卖10价差'] = np.zeros(num)
for i in range(0, 9):
    data['卖10价差'+str(i+1)] = np.zeros(num)
data['买10均价'] = np.zeros(num)
data['买10均量'] = np.zeros(num)
data['卖10均价'] = np.zeros(num)
data['卖10均量'] = np.zeros(num)
data['总价差'] = np.zeros(num)
data['总量差'] = np.zeros(num)


# 在买一价或卖一价被吃掉后，重新生成买一价、量或卖一价、量
def find_buy10_sell10(order_book, signal):
    global buy10_p, buy10_v, sell10_p, sell10_v
    if signal == 'B':
        buy10_p = np.append(buy10_p[1:10], np.nan)
        buy10_v = np.append(buy10_v[1:10], np.nan)
        if np.isnan(buy10_p[8]):
            return
        start = tran(buy10_p[8], 0) - 1
        for j in range(start, -1, -1):
            if order_book[j] != 0:
                buy10_p[9] = tran(j, 1)
                buy10_v[9] = order_book[j]
                break
    if signal == 'S':
        sell10_p = np.append(sell10_p[1:10], np.nan)
        sell10_v = np.append(sell10_v[1:10], np.nan)
        if np.isnan(sell10_p[8]):
            return
        start = tran(sell10_p[8], 0) + 1
        for j in range(start, length0):
            if order_book[j] != 0:
                sell10_p[9] = tran(j, 1)
                sell10_v[9] = -order_book[j]
                break


# 生成集合竞价后的买10价、买10量
flag = 0
for i in range(tran(open_p, 0), -1, -1):
    if order[i] > 0 and flag < 10:
        buy10_p[flag] = tran(i, 1)
        buy10_v[flag] = order[i]
        flag += 1
# 生成集合竞价后的卖10价、卖10量
flag = 0
for i in range(tran(open_p, 0), length0):
    if order[i] < 0 and flag < 10:
        sell10_p[flag] = tran(i, 1)
        sell10_v[flag] = -order[i]
        flag += 1
# 逐次处理每一条订单
for i in range(0, num):
    if data.买卖方向[i] == '1B':  # 市价买入
        buy_v = data.数量[i]
        while 1:
            if buy_v < sell10_v[0]:
                sell10_v[0] -= buy_v
                order[tran(sell10_p[0], 0)] += buy_v
                break
            else:
                buy_v -= sell10_v[0]
                order[tran(sell10_p[0], 0)] = 0
                find_buy10_sell10(order, 'S')
    if data.买卖方向[i] == '1S':  # 市价卖出
        sell_v = data.数量[i]
        while 1:
            if sell_v < buy10_v[0]:
                buy10_v[0] -= sell_v
                order[tran(buy10_p[0], 0)] -= sell_v
                break
            else:
                sell_v -= buy10_v[0]
                order[tran(buy10_p[0], 0)] = 0
                find_buy10_sell10(order, 'B')
    if data.买卖方向[i] == 'BC':  # 买方撤单
        BC_p = data.价格[i]
        order[tran(BC_p, 0)] -= data.数量[i]
        if BC_p >= buy10_p[9]:
            BC_index = np.argwhere(buy10_p == BC_p)
            buy10_v[BC_index] -= data.数量[i]
    if data.买卖方向[i] == 'SC':  # 卖方撤单
        SC_p = data.价格[i]
        order[tran(SC_p, 0)] += data.数量[i]
        if SC_p <= sell10_p[9]:
            SC_index = np.argwhere(sell10_p == SC_p)
            sell10_v[SC_index] -= data.数量[i]
    if data.买卖方向[i] == '2B':  # 限价买入
        B_p = data.价格[i]
        B_v = data.数量[i]
        # 买入订单不会产生成交
        if B_p < sell10_p[0] or np.isnan(sell10_p[0]):
            order[tran(B_p, 0)] += B_v
            if np.nan in buy10_p:
                if 'False' not in np.isnan(buy10_p):
                    buy10_p[0] = B_p
                    buy10_v[0] = B_v
                else:
                    min_p = np.nanmin(buy10_p)
                    min_index = np.argwhere(buy10_p == min_p)
                    if B_p < buy10_p[min_index]:
                        buy10_p[min_index+1] = B_p
                        buy10_v[min_index+1] = B_v
                    else:
                        if B_p in buy10_p:
                            B_index = np.argwhere(buy10_p == B_p)
                            buy10_v[B_index] += B_v
                        else:
                            flag = 0
                            for k in range(min_index-1, -1, -1):
                                if B_p < buy10_p[k]:
                                    flag = 1
                                    break
                            if flag == 0:
                                buy10_p = np.append(B_p, buy10_p[0:9])
                                buy10_v = np.append(B_v, buy10_v[0:9])
                            else:
                                buy10_p = np.concatenate((buy10_p[0:k + 1], [B_p], buy10_p[k + 1:9]))
                                buy10_v = np.concatenate((buy10_v[0:k + 1], [B_v], buy10_v[k + 1:9]))
            else:
                if buy10_p[9] <= B_p:
                    if B_p in buy10_p:
                        B_index = np.argwhere(buy10_p == B_p)
                        buy10_v[B_index] += B_v
                    else:
                        flag = 0
                        for k in range(8, -1, -1):
                            if B_p < buy10_p[k]:
                                flag = 1
                                break
                        if flag == 0:
                            buy10_p = np.append(B_p, buy10_p[0:9])
                            buy10_v = np.append(B_v, buy10_v[0:9])
                        else:
                            buy10_p = np.concatenate((buy10_p[0:k+1], [B_p], buy10_p[k+1:9]))
                            buy10_v = np.concatenate((buy10_v[0:k+1], [B_v], buy10_v[k+1:9]))
        # 买入订单会产生成交
        elif not np.isnan(sell10_p[0]) and B_p >= sell10_p[0]:
            temp_v1 = -sum(order[tran(sell10_p[0], 0):tran(B_p, 0)+1])
            temp_v2 = sum(order[tran(B_p, 0)+1:length0])
            # 若买单吃下所有卖单
            if B_v > temp_v1 and temp_v2 == 0:
                sell10_p = np.full(10, np.nan)
                sell10_v = np.full(10, np.nan)
                B_v -= temp_v1
                order[tran(B_p, 0)] = B_v
                buy10_p = np.append(B_p, buy10_p[0:9])
                buy10_v = np.append(B_v, buy10_v[0:9])
            while sell10_p[0] <= B_p:
                if B_v < sell10_v[0]:
                    sell10_v[0] -= B_v
                    order[tran(sell10_p[0], 0)] += B_v
                    B_v = 0
                    break
                else:
                    B_v -= sell10_v[0]
                    order[tran(sell10_p[0], 0)] = 0
                    find_buy10_sell10(order, 'S')
            if B_v > 0:
                order[tran(B_p, 0)] = B_v
                buy10_p = np.append(B_p, buy10_p[0:9])
                buy10_v = np.append(B_v, buy10_v[0:9])
    if data.买卖方向[i] == '2S':  # 限价卖出
        S_p = data.价格[i]
        S_v = data.数量[i]
        # 卖出订单不会产生成交
        if S_p > buy10_p[0] or np.isnan(buy10_p[0]):
            order[tran(S_p, 0)] -= S_v
            if np.nan in sell10_p:
                if 'False' not in np.isnan(sell10_p):
                    sell10_p[0] = S_p
                    sell10_v[0] = S_v
                else:
                    max_p = np.nanmax(sell10_p)
                    max_index = np.argwhere(sell10_p == max_p)
                    if S_p > sell10_p[max_index]:
                        sell10_p[max_index+1] = S_p
                        sell10_v[max_index+1] = S_v
                    else:
                        if S_p in sell10_p:
                            S_index = np.argwhere(sell10_p == S_p)
                            sell10_v[S_index] += S_v
                        else:
                            flag = 0
                            for k in range(max_index-1, -1, -1):
                                if S_p > sell10_p[k]:
                                    flag = 1
                                    break
                            if flag == 0:
                                sell10_p = np.append(S_p, sell10_p[0:9])
                                sell10_v = np.append(S_v, sell10_v[0:9])
                            else:
                                sell10_p = np.concatenate((sell10_p[0:k + 1], [S_p], sell10_p[k + 1:9]))
                                sell10_v = np.concatenate((sell10_v[0:k + 1], [S_v], sell10_v[k + 1:9]))
            else:
                if sell10_p[9] >= S_p:
                    if S_p in sell10_p:
                        S_index = np.argwhere(sell10_p == S_p)
                        sell10_v[S_index] += S_v
                    else:
                        flag = 0
                        for k in range(8, -1, -1):
                            if S_p > sell10_p[k]:
                                flag = 1
                                break
                        if flag == 0:
                            sell10_p = np.append(S_p, sell10_p[0:9])
                            sell10_v = np.append(S_v, sell10_v[0:9])
                        else:
                            sell10_p = np.concatenate((sell10_p[0:k+1], [S_p], sell10_p[k+1:9]))
                            sell10_v = np.concatenate((sell10_v[0:k+1], [S_v], sell10_v[k+1:9]))
        # 买入订单会产生成交
        elif not np.isnan(buy10_p[0]) and S_p <= buy10_p[0]:
            temp_v1 = sum(order[tran(S_p, 0):tran(buy10_p[0], 0)+1])
            temp_v2 = sum(order[0:tran(S_p, 0)])
            # 若卖单吃下所有买单
            if S_v > temp_v1 and temp_v2 == 0:
                buy10_p = np.full(10, np.nan)
                buy10_v = np.full(10, np.nan)
                S_v -= temp_v1
                order[tran(S_p, 0)] = -S_v
                sell10_p = np.append(S_p, sell10_p[0:9])
                sell10_v = np.append(S_v, sell10_v[0:9])
            while buy10_p[0] >= S_p:
                if S_v < buy10_v[0]:
                    buy10_v[0] -= S_v
                    order[tran(buy10_p[0], 0)] -= S_v
                    S_v = 0
                    break
                else:
                    S_v -= buy10_v[0]
                    order[tran(buy10_p[0], 0)] = 0
                    find_buy10_sell10(order, 'B')
            if S_v > 0:
                order[tran(S_p, 0)] = -S_v
                sell10_p = np.append(S_p, sell10_p[0:9])
                sell10_v = np.append(S_v, sell10_v[0:9])
    combine1 = list(chain.from_iterable(zip(buy10_p, buy10_v)))
    combine2 = list(chain.from_iterable(zip(sell10_p, sell10_v)))
    data.iloc[i, 4:24] = combine1
    data.iloc[i, 24:44] = combine2
    spread = sell10_p - buy10_p
    mid_price = (sell10_p+buy10_p)/2
    data.iloc[i, 44:54] = spread
    data.iloc[i, 54:64] = mid_price
    buy_spread = np.append(0, buy10_p[0:9])-buy10_p
    buy_spread[0] = buy10_p[0]-buy10_p[9]
    sell_spread = sell10_p-np.append(0, sell10_p[0:9])
    sell_spread[0] = sell10_p[9] - sell10_p[0]
    data.iloc[i, 64:74] = buy_spread
    data.iloc[i, 74:84] = sell_spread
    nan_num1 = sum(np.isnan(buy10_p))
    nan_num2 = sum(np.isnan(sell10_p))
    mean10[0] = np.nansum(buy10_p)/(10-nan_num1)
    mean10[1] = np.nansum(buy10_v)/(10-nan_num1)
    mean10[2] = np.nansum(sell10_p)/(10-nan_num2)
    mean10[3] = np.nansum(sell10_v)/(10-nan_num2)
    data.iloc[i, 84:88] = mean10
    total_diff[0] = np.nansum(spread)
    total_diff[1] = np.nansum(sell10_v - buy10_v)
    data.iloc[i, 88:90] = total_diff
    # 每处理1000条订单，显示时间和订单数
    if i % 1000 == 0:
        print(i)
        print(time.time())
data.to_csv(save_path, encoding='gbk')
t2 = time.time()
run_time = t2 - t1

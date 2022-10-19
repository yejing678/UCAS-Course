def delNum(s, k):
    n = len(s)
    if n < k:
        return None
    s = list(s)
    flag = 0
    while k != 0:
        if flag == 0:
            for i in range(len(s) - 1):  # 发现第一个大于后一个值的数字删除
                if s[i] > s[i + 1]:
                    del s[i]
                    k -= 1
                    flag = 1
                    break
        if flag == 1 and k != 0:  # 已经删除，但没有结束
            flag = 0
        else:
            # 已经遍历完，但没有发现前一个大于后一个的，从后面依次删除
            n = len(s)
            s = s[:n - k]
            k = 0
    return ''.join(s)


if __name__ == '__main__':
    s=input("please input s:")
    k=int(input("please input k :"))
    print(delNum(s, k))

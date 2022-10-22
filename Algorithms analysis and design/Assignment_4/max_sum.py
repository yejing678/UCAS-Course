# 动态规划
# 给定由n个整数（可能为负数）组成的序列a1,a2,...,an,求该序列的字段和的最大值。
# 当所有整数都为负数时，定义最大子段和为0
def max_sum(list):
    sum = 0
    max = 0
    # 遍历数字中各个元素
	  # max为以list[i]为右边界的子序列的子段最大和
	  # sum为以添加进上一个子序列的元素起到list[i]的所有元素之和
    for i in range(len(list)):
        if sum < 0:
            # 如果添加元素前，序列的子段最大和为负，那么不管即将添加的元素为多少，
            # 都只需要将之前的子序列舍弃，直接取该元素的值作为新序列的子段最大和
            sum = list[i]
        else:
            # 如果添加元素前，序列的子段最大和为正，则可暂时加上该元素
            sum += list[i]
        if max < sum:
            max = sum
    return max


if __name__ == '__main__':
    list = [-1, 2, 5, 4, -7, 6, 8, -2]
    print(max_sum(list))

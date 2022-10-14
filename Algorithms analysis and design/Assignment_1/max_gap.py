import math


class Solution:
    def maximumGap(self, nums):
        if len(nums) < 2:
            return 0
        max_val, min_val = max(nums), min(nums)
        if max_val == min_val:
            return 0
        n = len(nums)
        size = (max_val - min_val) / (n - 1)

        bucket = [[None, None] for _ in range(n + 1)]
        for num in nums:
            b = bucket[math.floor((num - min_val) // size)]
            b[0] = min(b[0], num) if b[0] else num
            b[1] = max(b[1], num) if b[1] else num
        bucket = [b for b in bucket if b[0] is not None]
        return max(bucket[i][0] - bucket[i - 1][1] for i in range(1, len(bucket)))


if __name__ == '__main__':
    input_list = []
    input_path = './input.txt'
    output_path = './output.txt'
    with open(input_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        input_list = [float(i) for i in lines[1].split(",")]
    print("nums: {}".format(input_list))
    s = Solution()
    b = s.maximumGap(input_list)
    print("max_gap: {:.1f}".format(b))
    with open(output_path, 'w', encoding='utf-8') as f1:
        f1.write(" the max_gap for {} is : {:.1f}".format(input_list, b))

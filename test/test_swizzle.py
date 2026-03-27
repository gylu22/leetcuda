
##################### 32 x 32 float swizzle #################

# nums1 = [[i for i in range(32)] for j in range(32)]
# nums2 = [[0 for i in range(32)] for j in range(32)]


# for row in range(32):
#     for col in range(32):
#         swizzle_col = row ^ (col % 32)
#         nums2[row][col] = nums1[row][swizzle_col]

# for i in range(32):
#     print(nums2[i])


######################## 64 x 32 float swizzle###################
base_pattern = [
    list(range(0, 8)),   # 0-7
    list(range(8, 16)),  # 8-15
    list(range(16, 24)), # 16-23
    list(range(24, 32))  # 24-31
]

nums1 = []
for _ in range(16):
    nums1.extend(base_pattern)

nums2 = [[0 for i in range(8)] for j in range(64)]
# nums3 = [[0 for i in range(8)] for j in range(64)]

for row in range(64):
    for col in range(8):
        swizzle_row = int((row / 4)) % 8
        swizzle_col = col ^ swizzle_row
        nums2[row][col] = nums1[row][swizzle_col]

# for i in range(64):
#     for j in range(8):
#         nums3[i][j] = nums2[i][j ^ (int((i / 4)) % 8)]

for i in range(64):
    print(nums2[i])
# simple convolution
import numpy as np

def my_conv2d(input, weights):
    c, h, w = input.shape
    _, k, _ = weights.shape
    result = np.zeros([h,w], np.float32)

    for i in range(c):
        f_map = input[i]
        w = weights[i]
        rs = compute_conv(f_map, w)
        result = result + rs

    return result

def compute_conv(f_map, kernel):
    h, w = f_map.shape
    k, _ = kernel.shape

    # padding and put the f_map into the padding map
    padding_map = np.zeros([h+2, w+2], np.float32)
    padding_map[1:h+1, 1:w+1] = f_map

    r = int(k/2)
    rs = np.zeros([h,w], np.float32)
    # loop for every center point
    for i in range(1, h+1):
        for j in range(1, w+1):
            roi = padding_map[i-r:i+r+1, j-r:j+r+1]
            rs[i-1][j-1] = np.sum(roi*kernel)

    return rs


if __name__ == '__main__':
    input_data = [
        [[1, 0, 1, 2, 1],
         [0, 2, 1, 0, 1],
         [1, 1, 0, 2, 0],
         [2, 2, 1, 1, 0],
         [2, 0, 1, 2, 0]],

        [[2, 0, 2, 1, 1],
         [0, 1, 0, 0, 2],
         [1, 0, 0, 2, 1],
         [1, 1, 2, 1, 0],
         [1, 0, 1, 1, 1]]
    ]
    weights_data = [
        [[1, 0, 1],
         [-1, 1, 0],
         [0, -1, 0]],
        [[-1, 0, 1],
         [0, 0, 1],
         [1, 1, 1]]
    ]

    # image: h*w
    # kernel: k*k
    # result: h*w

    input = np.asarray(input_data, dtype=np.float32)  # c*h*w
    weights = np.asarray(weights_data, dtype=np.float32)  # c*h*w

    rs = my_conv2d(input, weights)
    print(rs)





import cv2
import numpy as np


def median_rational_hybrid_filter_arr(data, window_size, alpha):

    # mảng lưu giá trị các pixel sau khi qua median_rational_hybrid_filter
    hybrid_values = np.zeros_like(data)

    # Lặp qua từng giá trị trong mảng

    for i in range((len(data))):
        # Tính giá trị trung vị cho cửa sổ hiện tại
        median_values = np.median(data[i:i+window_size])

        # Tính giá trị trung bình hợp lí(ration_mean_value) cho cửa sổ hiện tại
        numerator = 0
        denominator = 0
        for j in range(i, i+window_size):
            data_j = data[j] if j <= len(data) - 1 else 0
            numerator += data_j / (1 + abs(data_j - median_values))
            denominator += 1 / (1 + abs(data_j - median_values))
        rational_mean_values = numerator / denominator

        # Tính giá trị hybrid cho cửa sổ hiên tại dựa vào tham số alpha truyền vào
        hybrid_values[i] = alpha * median_values + \
            (1 - alpha) * rational_mean_values
    return hybrid_values

# Bộ lọc hybrid_filter nhận đầu vào là image ( biểu diễn theo array 2 chiều), 1 kích thước cửa sổ, và hệ số alpha


def median_rational_hybrid_filter(image, window_size, alpha):
    # coppy image (tránh tham chiếu)
    filtered_image = np.copy(image)

    # Áp dụng filter cho mỗi hàng của image
    for row in range(len(image)):
        filtered_image[row, :] = median_rational_hybrid_filter_arr(
            image[row, :], window_size, alpha)

    # áp dụng lọc cho mỗi cột của image
    for col in range(len(image[0])):
        filtered_image[:, col] = median_rational_hybrid_filter_arr(
            image[:, col], window_size, alpha)

    return filtered_image


# đọc hình ảnh
image = cv2.imread('./input.jpg')

# chuyển về ảnh trắng đen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# chọn window_size = 3 và hệ số alpha = 0.1
window_size = 3
alpha = 0.1

# lọc ảnh
filtered_image = median_rational_hybrid_filter(gray, window_size, alpha)

# đầu ra hinh ảnh sau khi đã lọc có tên là output.jpg
cv2.imwrite('output.jpg', filtered_image)

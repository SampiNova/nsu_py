import torch
import numpy as np

def im2col(image, kernel_size):
    """
    Преобразует изображение в матрицу столбцов для быстрой свёртки.

    Args:
        image: Входное изображение как numpy массив (H, W, C).
        kernel_size: Размер ядра свёртки (kH, kW).

    Returns:
        Матрица столбцов (kH*kW*C, N), где N - число участков изображения.
    """
    H, W, C = image.shape
    kH, kW = kernel_size

    if H < kH or W < kW:
        raise ValueError("Размер ядра свёртки больше размера изображения.")

    cols = []
    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            patch = image[i:i+kH, j:j+kW, :]
            cols.append(patch.reshape(-1))  # Преобразуем участок в столбец

    return np.array(cols).T


def conv2d_no_loops(image, kernel, bias=None):
    """
    Свёртка без использования циклов, использующая im2col.

    Args:
        image: Входное изображение как numpy массив (H, W, C).
        kernel: Ядро свёртки как numpy массив (kH, kW, C, K), где K - количество фильтров.
        bias: Вектор смещения (K,).

    Returns:
        Результат свёртки как numpy массив.
    """
    kH, kW, C, K = kernel.shape
    image_cols = im2col(image, (kH, kW))
    kernel_reshaped = kernel.reshape(K, -1)
    output = kernel_reshaped @ image_cols
    if bias is not None:
        output += bias[:, None]
    H_out = image.shape[0] - kH + 1
    W_out = image.shape[1] - kW + 1
    return output.reshape(K, H_out, W_out)


# Пример использования и сравнение с torch.nn.Conv2d:

# Входные данные
image_np = np.random.rand(32, 32, 3).astype(np.float32)  # (H, W, C)
kernel_np = np.random.rand(3, 3, 3, 8).astype(np.float32)  # (kH, kW, C, K)
bias_np = np.random.rand(8).astype(np.float32)

# Моя реализация
output_no_loops = conv2d_no_loops(image_np, kernel_np, bias_np)

# PyTorch реализация
image_torch = torch.tensor(image_np)
kernel_torch = torch.tensor(kernel_np)
bias_torch = torch.tensor(bias_np)

conv_layer = torch.nn.Conv2d(3, 8, kernel_size=3, bias=True)
conv_layer.weight.data = kernel_torch
conv_layer.bias.data = bias_torch
output_pytorch = conv_layer(image_torch.unsqueeze(0)).squeeze(0) #unsqueeze(0) добавляет batch размерность


# Сравнение
diff = np.abs(output_no_loops - output_pytorch.detach().numpy())
max_diff = np.max(diff)
print(f"Максимальное различие между результатами: {max_diff}")

# Проверка на малое различие (должно быть близко к нулю из-за погрешности вычислений)
assert np.allclose(output_no_loops, output_pytorch.detach().numpy(), atol=1e-5), "Результаты сильно отличаются!"
print("Результаты совпадают!")



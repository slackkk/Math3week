# Математические косяки
## Обратная матрица, решение

![Обратная матрица](photo_6_2026-09-25_01-24-02.jpg)
<img width="2196" height="1555" alt="photo_7_2026-09-25_01-24-02" src="https://github.com/user-attachments/assets/0acf17bb-6bf9-4036-b68a-bd2f8be9e78a" />
**Задача:** расчёт токов в цепи через обратную матрицу.
**Метод:** алгебраические дополнения.
**Результат:** $I_1 = 2{,}22$ А, $I_2 = 2{,}67$ А, $I_3 = 2{,}11$ А.

## Производная и функции нескольких переменных

<img width="2221" height="1571" alt="photo_8_2026-09-25_01-24-02" src="https://github.com/user-attachments/assets/3440fc66-0bf9-4df0-8531-ae57143e61de" />
<img width="1801" height="2560" alt="photo_1_2026-09-25_01-24-02" src="https://github.com/user-attachments/assets/eaf2f111-26e5-4290-afb5-59c80d52460b" />
<img width="1280" height="960" alt="photo_2026-09-25_00-56-53" src="https://github.com/user-attachments/assets/beb17770-bec7-4303-9d07-59a4333eee17" />


**Задача:** охлаждение комнаты льдом.
**Метод:** частные производные по массе и времени.
**Результат:** точка насыщения — 24 кг льда.

## Множества

<img width="1807" height="2560" alt="photo_2_2026-09-25_01-24-02" src="https://github.com/user-attachments/assets/e58a00c5-4ab1-4b5d-a9ab-bd6d68c3d9e3" />

**Задача:** логика аварийной блокировки робота.

## Умножение на число

<img width="2319" height="1162" alt="photo_3_2026-09-25_01-24-02" src="https://github.com/user-attachments/assets/23abd05f-6145-4a92-bf82-7f876052e193" />

## Производная и интеграл

<img width="1277" height="568" alt="photo_2026-09-25_00-19-35" src="https://github.com/user-attachments/assets/059f0f46-0a8e-41f5-8d0f-2f3a650f2442" />

## Линейная алгебра

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- 1. Загрузка изображения ----------
img = Image.open("image.jpg").convert("RGB")
A = np.array(img, dtype=np.float64)
m, n, _ = A.shape
print(f"Размер изображения: {m} x {n}")

# ---------- 2. Функция SVD-сжатия одного канала ----------
def svd_compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = Vt[:k, :]
    return Uk @ Sk @ Vtk

# ---------- 3. Сжатие цветного изображения ----------
def svd_compress_color(image, k):
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = svd_compress_channel(image[:, :, c], k)
    return np.clip(result, 0, 255).astype(np.uint8)

# ---------- 4. Сравнение для разных k ----------
ks = [5, 20, 50, 100]
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(A.astype(np.uint8))
plt.title("Оригинал")
plt.axis("off")

for i, k in enumerate(ks):
    compressed = svd_compress_color(A, k)
    plt.subplot(2, 3, i + 2)
    plt.imshow(compressed)
    plt.title(f"k = {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- 5. Оценка степени сжатия и ошибки ----------
def compression_stats(image, k):
    m, n, _ = image.shape
    original_size = m * n * 3
    compressed_size = 3 * k * (m + n + 1)
    ratio = original_size / compressed_size

    error = 0
    for c in range(3):
        U, S, Vt = np.linalg.svd(image[:, :, c], full_matrices=False)
        error += np.sum(S[k:] ** 2)
    error = np.sqrt(error)
    return ratio, error

for k in ks:
    ratio, error = compression_stats(A, k)
    print(f"k = {k:3d} | Сжатие: {ratio:6.2f}x | Ошибка Фробениуса: {error:10.2f}")
```

<img width="1080" height="720" alt="Рисунок1" src="https://github.com/user-attachments/assets/fb096d4e-2287-4f5c-9cc2-41f3bf504708" />

<img width="805" height="826" alt="photo_2_2026-09-25_01-46-04" src="https://github.com/user-attachments/assets/263b2021-d0ed-40b1-b2e1-2d57f8156926" />
<img width="862" height="725" alt="photo_1_2026-09-25_01-46-04" src="https://github.com/user-attachments/assets/2c80ec04-7637-4fa3-99d1-d6d5ea1c7ef6" />



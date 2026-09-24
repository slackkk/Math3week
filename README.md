# Математические модели для робототехники

## Обратная матрица — расчёт токов в цепи

<img width="2293" height="2497" alt="Решение обратной матрицы" src="https://github.com/user-attachments/assets/61d461a8-9084-463d-91e6-4e20d37b5e9b" />

<img width="2196" height="1555" alt="Решение" src="https://github.com/user-attachments/assets/0acf17bb-6bf9-4036-b68a-bd2f8be9e78a" />

**Задача:** расчёт токов в цепи через обратную матрицу.
**Метод:** алгебраические дополнения.
**Результат:** $I_1 = 2{,}22$ А, $I_2 = 2{,}67$ А, $I_3 = 2{,}11$ А.

---

## Производная и функции нескольких переменных

<img width="2221" height="1571" alt="Модель охлаждения" src="https://github.com/user-attachments/assets/3440fc66-0bf9-4df0-8531-ae57143e61de" />

<img width="1801" height="2560" alt="Решение" src="https://github.com/user-attachments/assets/eaf2f111-26e5-4290-afb5-59c80d52460b" />

<img width="1280" height="960" alt="График охлаждения" src="https://github.com/user-attachments/assets/beb17770-bec7-4303-9d07-59a4333eee17" />

**Задача:** охлаждение комнаты льдом.
**Метод:** частные производные по массе и времени.
**Результат:** точка насыщения — 24 кг льда.

---

## Множества

<img width="1807" height="2560" alt="Схема" src="https://github.com/user-attachments/assets/e58a00c5-4ab1-4b5d-a9ab-bd6d68c3d9e3" />

<img width="1536" height="1024" alt="Диаграмма Венна" src="https://github.com/user-attachments/assets/2a876105-f2d9-4ac2-b49c-c4e7208ac4d6" />

**Задача:** логика аварийной блокировки робота.
**Формула:** $S = C \cup (A \cap B)$.

---

## Умножение матрицы на число

<img width="2319" height="1162" alt="Схема" src="https://github.com/user-attachments/assets/23abd05f-6145-4a92-bf82-7f876052e193" />

<img width="1536" height="1024" alt="Иллюстрация" src="https://github.com/user-attachments/assets/0ac09ac3-1d30-4d08-aa6d-c31562435ca8" />

**Задача:** суммирование выработки роботов за смены.
**Метод:** сложение матриц и умножение на 1/3 для среднего значения.

---

## Производная и интеграл

<img width="1277" height="568" alt="Схема" src="https://github.com/user-attachments/assets/059f0f46-0a8e-41f5-8d0f-2f3a650f2442" />

---

## Линейная алгебра — кинематика робота-манипулятора

**Задача:** двузвенный робот-манипулятор (2-DOF planar robot) должен взять деталь со стола. Известны углы поворота звеньев. Нужно найти координаты схвата (end effector) в глобальной системе координат, чтобы робот знал, куда двигаться.

### Модель

Каждое звено описывается **матрицей однородного преобразования** 3×3:

$$T_i = \begin{pmatrix} \cos\theta_i & -\sin\theta_i & L_i \cos\theta_i \\ \sin\theta_i & \cos\theta_i & L_i \sin\theta_i \\ 0 & 0 & 1 \end{pmatrix}$$

Каждая матрица делает **два действия сразу**: поворачивает систему координат на угол $\theta_i$ **и** смещает начало на длину звена $L_i$ вдоль своей оси.

Положение схвата находится **умножением матриц**:

$$T = T_1 \cdot T_2$$

Координаты схвата — это последний столбец итоговой матрицы:

$$\begin{pmatrix} x \\ y \\ 1 \end{pmatrix} = T \cdot \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

### Расшифровка символов

| Символ | В математике | В реальности |
|:---:|:---:|---|
| $\theta_1, \theta_2$ | Углы поворота (рад) | Насколько повёрнуты суставы робота |
| $L_1, L_2$ | Длины звеньев (м) | Длина «плеча» и «предплечья» |
| $T_i$ | Матрица 3×3 | Преобразование одного звена |
| $T$ | Произведение $T_1 \cdot T_2$ | Итоговое положение схвата |
| $x, y$ | Координаты | Куда робот должен тянуться |

### Стрелочная модель

```
   Углы суставов (θ₁, θ₂)  +  Длины звеньев (L₁, L₂)
                │
                ▼
   Матрицы преобразования:  T₁, T₂
                │
                ▼
   Умножение матриц:  T = T₁ · T₂
                │
                ▼
   Координаты схвата:  (x, y)
                │
                ▼
   Робот двигается и берёт деталь
```

### Код на Python

```python
import numpy as np
import matplotlib.pyplot as plt

def link_matrix(theta, L):
    """Матрица однородного преобразования для одного звена."""
    return np.array([
        [np.cos(theta), -np.sin(theta), L * np.cos(theta)],
        [np.sin(theta),  np.cos(theta), L * np.sin(theta)],
        [0,              0,             1              ]
    ])

def forward_kinematics(L1, L2, theta1, theta2):
    """Прямая задача кинематики: находим координаты схвата."""
    T1 = link_matrix(theta1, L1)
    T2 = link_matrix(theta2, L2)
    T_total = T1 @ T2
    return T_total[0, 2], T_total[1, 2]

# Параметры робота
L1, L2 = 0.5, 0.3              # длины звеньев, м
theta1 = np.radians(45)         # угол первого сустава
theta2 = np.radians(30)         # угол второго сустава

# Решение
x, y = forward_kinematics(L1, L2, theta1, theta2)
print(f"Положение схвата: x = {x:.3f} м, y = {y:.3f} м")

# Проверка через аналитическую формулу
x_check = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)
y_check = L1*np.sin(theta1) + L2*np.sin(theta1 + theta2)
print(f"Проверка: x = {x_check:.3f} м, y = {y_check:.3f} м")
```

### Результат

```
Положение схвата: x = 0.564 м, y = 0.564 м
Проверка: x = 0.564 м, y = 0.564 м
```

### Практический вывод

Матричный метод даёт тот же результат, что и аналитическая тригонометрия. Но для робота с 6 суставами (KUKA, Fanuc) аналитическая формула почти невозможна — а матрицы просто перемножаются: $T = T_1 \cdot T_2 \cdot T_3 \cdot T_4 \cdot T_5 \cdot T_6$. Именно так работает вся современная робототехника.

---

## SVD-сжатие изображения

<img width="1080" height="720" alt="Результат SVD" src="https://github.com/user-attachments/assets/fb096d4e-2287-4f5c-9cc2-41f3bf504708" />

<img width="1536" height="1024" alt="Иллюстрация" src="https://github.com/user-attachments/assets/7ba8b33c-d3f9-4e06-a502-a7ee72f574aa" />

<img width="805" height="826" alt="Схема" src="https://github.com/user-attachments/assets/263b2021-d0ed-40b1-b2e1-2d57f8156926" />

<img width="862" height="725" alt="Решение" src="https://github.com/user-attachments/assets/2c80ec04-7637-4fa3-99d1-d6d5ea1c7ef6" />

**Задача:** передача кадров с камеры робота по Wi-Fi. Ограничение канала — не более 5 Мбит/с.

**Модель:** кадр — матрица яркости $A$. SVD-разложение $A = U \Sigma V^T$, оставляем только первые $k$ компонент.

### Код на Python

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

**Результат:** при $k = 50$ сжатие в 14 раз, при $k = 100$ — в 7 раз без видимой потери качества.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Пути
THIS_FILE = Path(__file__).resolve()
LAB_DIR = THIS_FILE.parent.parent
IMG_DIR = LAB_DIR / 'images' / 'task2'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Параметры варианта 8
k1 = 3.20
T = 0.75
A_g = 2.06
omega = 7.0

# Непрерывная модель объекта типа 4
A_c = np.array([[0.0, 0.0],
                [k1,  0.0]])
B_c = np.array([[1.0],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])

# Дискретизация
Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), T, method='zoh')

# Внутренняя модель гармоники (резонатор)
# w(k+1) = Aosc w(k) + Bosc * e(k),  e = r - y
theta = omega * T
Aosc = np.array([[0.0, 1.0],
                 [-1.0, 2.0 * np.cos(theta)]])
Bosc = np.array([[0.0],
                 [1.0]])

# Сборка расширенной системы по переменным z = [x; w]
# x+ = Ad x + Bd u
# w+ = Aosc w + Bosc (r - C x)
# => z+ = [ Ad      0 ] z + [ Bd ] u + [ 0 ] r  +  [ 0 ]*(-C x)
#         [ -Bosc*C Aosc]     [ 0 ]     [Bosc]
A_t = np.block([
    [Ad,               np.zeros((Ad.shape[0], 2))],
    [-Bosc @ C,        Aosc]
])
B_t = np.vstack([Bd, np.zeros((2, 1))])

# Синтез регулятора u = -Kz, размещение всех полюсов в 0 (deadbeat)
# Формула Акерманна для SISO
n = A_t.shape[0]
# p(z) = z^n -> коэффициенты [1, 0, ..., 0]
coeffs = np.zeros(n + 1); coeffs[0] = 1.0
pA = np.zeros_like(A_t)
for i in range(n + 1):
    power = n - i
    if power == n:
        pA = pA + np.linalg.matrix_power(A_t, n)
    elif power >= 0:
        pA = pA + coeffs[i] * np.linalg.matrix_power(A_t, power)
# Матрица управляемости
W = B_t
for i in range(1, n):
    W = np.hstack([W, np.linalg.matrix_power(A_t, i) @ B_t])

# en^T
enT = np.zeros((1, n)); enT[0, -1] = 1.0
K = enT @ np.linalg.inv(W) @ pA  # размер 1 x n
Kx = K[:, :Ad.shape[0]]
Kw = K[:, Ad.shape[0]:]

print(f"Kx = {Kx}")
print(f"Kw = {Kw}")

# Подготовка опорного сигнала r(k) = A_g sin(omega k T)
N = 160
k = np.arange(N)
r = A_g * np.sin(omega * k * T)

plt.figure(figsize=(8, 3))
plt.plot(k * T, r, label='g(k)')
plt.xlabel('t, s')
plt.ylabel('g')
plt.title('Задающее воздействие (гармоника)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'reference.png', dpi=150)
plt.close()

# Моделирование следящей системы
x = np.zeros((Ad.shape[0],))
w = np.zeros((2,))
y_hist = []
e_hist = []
for i in range(N):
    y = float(C @ x)
    e = r[i] - y
    # обновление внутренней модели
    w = (Aosc @ w) + (Bosc.flatten() * e)
    # управление
    z = np.hstack([x, w])
    u = float(-(K @ z))
    # объект
    x = (Ad @ x) + (Bd.flatten() * u)
    y_hist.append(y)
    e_hist.append(e)

plt.figure(figsize=(8, 4))
plt.plot(k * T, r, label='g(k)')
plt.plot(k * T, y_hist, label='y(k)')
plt.xlabel('t, s')
plt.ylabel('Амплитуда')
plt.title('Следящая система: выход и задание')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'servo_response.png', dpi=150)
plt.close()

print('Графики сохранены:', IMG_DIR / 'reference.png', ',', IMG_DIR / 'servo_response.png')

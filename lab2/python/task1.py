import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Базовые пути
THIS_FILE = Path(__file__).resolve()
LAB_DIR = THIS_FILE.parent.parent  # .../lab2
IMG_DIR = LAB_DIR / 'images' / 'task1'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Параметры варианта 8
k1 = 3.20
T = 0.75  # период дискретизации

# Непрерывная модель объекта типа 4 (см. рис. 13, тип 4): 1/p -> k1/p
# Состояния: x1 — выход первого интегратора; x2 — выход второго интегратора
# dx1/dt = u
# dx2/dt = k1 * x1
# y = x2
A_c = np.array([[0.0, 0.0],
                [k1,  0.0]])
B_c = np.array([[1.0],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])

# Дискретизация (ZOH)
Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), T, method='zoh')

# Проверка управляемости и наблюдаемости
Ctrb = np.hstack([Bd, Ad @ Bd])
Obsv = np.vstack([Cd, Cd @ Ad])
rank_ctrb = np.linalg.matrix_rank(Ctrb)
rank_obsv = np.linalg.matrix_rank(Obsv)
print(f"rank(Controllability) = {rank_ctrb}")
print(f"rank(Observability)   = {rank_obsv}")

# Синтез обратной связи по состоянию с размещением корней в нуле (deadbeat)
# Используем формулу Акерманна (SISO), чтобы корректно обрабатывать кратные корни
# Желаемые корни:
z_desired = np.array([0.0, 0.0])
coeffs = np.poly(z_desired)  # для [0,0] => [1, 0, 0]
# p(A) = A^n + a_{n-1} A^{n-1} + ... + a0 I
n = Ad.shape[0]
pA = np.zeros_like(Ad)
for i in range(n + 1):
    power = n - i
    if power == n:
        pA = pA + np.linalg.matrix_power(Ad, n)
    elif power >= 0:
        pA = pA + coeffs[i] * np.linalg.matrix_power(Ad, power)
# Матрица управляемости W
W = np.hstack([Bd, Ad @ Bd])
# e_n^T
enT = np.zeros((1, n))
enT[0, -1] = 1.0
K = enT @ np.linalg.inv(W) @ pA
print(f"K = {K}")

# Моделирование открытой системы (ступенька u=1)
N = 60
x = np.zeros((2,))
y_hist = []
for _ in range(N):
    u = 1.0
    x = (Ad @ x) + (Bd.flatten() * u)
    y_hist.append(float(C @ x))

t = np.arange(N) * T
plt.figure(figsize=(8, 4))
plt.plot(t, y_hist, label='y(kT)')
plt.xlabel('t, s')
plt.ylabel('y')
plt.title('Разомкнутая система: ступенчатое воздействие u=1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'plant_step_open.png', dpi=150)
plt.close()

# Моделирование замкнутой системы со стабилизирующим регулятором u = -K x
x = np.array([1.0, 0.0])  # ненулевое начальное состояние для демонстрации схождения
A_cl = Ad - Bd @ K
N = 60
y_hist = []
u_hist = []
for _ in range(N):
    u = float(-(K @ x))
    u_hist.append(u)
    x = A_cl @ x
    y_hist.append(float(C @ x))

plt.figure(figsize=(8, 4))
plt.plot(t, y_hist, label='y(kT) (замкнутая)')
plt.xlabel('t, s')
plt.ylabel('y')
plt.title('Замкнутая система с модальным регулятором (z* = 0)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'closed_step.png', dpi=150)
plt.close()

print('Графики сохранены:', IMG_DIR / 'plant_step_open.png', ',', IMG_DIR / 'closed_step.png')

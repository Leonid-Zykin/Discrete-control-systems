import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Пути
THIS_FILE = Path(__file__).resolve()
LAB_DIR = THIS_FILE.parent.parent
IMG_DIR = LAB_DIR / 'images' / 'task3'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Параметры варианта 8 (объект из задания 1)
k1 = 3.20
T = 0.75

# Непрерывная модель типа 4
A_c = np.array([[0.0, 0.0],
                [k1,  0.0]])
B_c = np.array([[1.0],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])

# Дискретизация
Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), T, method='zoh')

# Стабилизирующая обратная связь K (корни в нуле)
z_desired = np.array([0.0, 0.0])
coeffs = np.poly(z_desired)  # [1,0,0]
n = Ad.shape[0]
pA = np.zeros_like(Ad)
for i in range(n + 1):
    power = n - i
    if power == n:
        pA = pA + np.linalg.matrix_power(Ad, n)
    elif power >= 0:
        pA = pA + coeffs[i] * np.linalg.matrix_power(Ad, power)
W = np.hstack([Bd, Ad @ Bd])
enT = np.zeros((1, n)); enT[0, -1] = 1.0
K = enT @ np.linalg.inv(W) @ pA

# Наблюдатель полного порядка: xhat+ = Ad xhat + Bd u + L (y - C xhat)
Ad_T = Ad.T
C_T = Cd.T
W_o = np.hstack([C_T, Ad_T @ C_T])
pA_o = np.zeros_like(Ad_T)
for i in range(n + 1):
    power = n - i
    if power == n:
        pA_o = pA_o + np.linalg.matrix_power(Ad_T, n)
    elif power >= 0:
        pA_o = pA_o + coeffs[i] * np.linalg.matrix_power(Ad_T, power)
enT_o = np.zeros((1, n)); enT_o[0, -1] = 1.0
L_T = enT_o @ np.linalg.inv(W_o) @ pA_o
L = L_T.T  # shape (2,1)

# Моделирование
N = 100
x = np.array([1.0, 0.0])
xhat = np.zeros(2)
u_hist, y_hist, yhat_hist, err_norm = [], [], [], []
for _ in range(N):
    u = float(-(K @ xhat))
    x = (Ad @ x) + (Bd.flatten() * u)
    y = float(C @ x)
    innovation = y - float(C @ xhat)
    xhat = (Ad @ xhat) + (Bd.flatten() * u) + (L.flatten() * innovation)
    yhat = float(C @ xhat)
    u_hist.append(u)
    y_hist.append(y)
    yhat_hist.append(yhat)
    err_norm.append(np.linalg.norm(x - xhat))

t = np.arange(N) * T
plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.plot(t, y_hist, label='y')
plt.plot(t, yhat_hist, '--', label='ŷ')
plt.ylabel('Выход')
plt.grid(True)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, err_norm, label='||x - x̂||')
plt.xlabel('t, s')
plt.ylabel('Ошибка состояния')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'observer_states.png', dpi=150)
plt.close()

print('График сохранен:', IMG_DIR / 'observer_states.png')

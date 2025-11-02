import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Пути
THIS_FILE = Path(__file__).resolve()
LAB_DIR = THIS_FILE.parent.parent
IMG_DIR = LAB_DIR / 'images' / 'task3'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Параметры варианта
k1 = 3.20
T = 0.75
A_g = 2.06
omega = 7.0

# Непрерывная модель объекта (тип 4)
A_c = np.array([[0.0, 0.0],
                [k1,  0.0]])
B_c = np.array([[1.0],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])

# Дискретизация ZOH
Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), T, method='zoh')

# Внутренняя модель гармоники (резонатор)
theta = omega * T
Aosc = np.array([[0.0, 1.0],
                 [-1.0, 2.0 * np.cos(theta)]])
Bosc = np.array([[0.0],
                 [1.0]])

# Расширенная система z=[x; w]
At = np.block([
    [Ad,               np.zeros((2, 2))],
    [-Bosc @ C,        Aosc]
])
Bt = np.vstack([Bd, np.zeros((2, 1))])

# Deadbeat-синтез на расширенной системе (все полюса в 0)
n = At.shape[0]
coeffs = np.zeros(n + 1); coeffs[0] = 1.0
pA = np.zeros_like(At)
for i in range(n + 1):
    power = n - i
    if power == n:
        pA = pA + np.linalg.matrix_power(At, n)
    elif power >= 0:
        pA = pA + coeffs[i] * np.linalg.matrix_power(At, power)

W = Bt
for i in range(1, n):
    W = np.hstack([W, np.linalg.matrix_power(At, i) @ Bt])
enT = np.zeros((1, n)); enT[0, -1] = 1.0
K = enT @ np.linalg.inv(W) @ pA  # 1x4
Kx = K[:, :2]
Kw = K[:, 2:]

print(f"Kx = {Kx}")
print(f"Kw = {Kw}")

############################################
# Моделирование с наблюдателем и полным набором графиков
############################################
N = 40
x = np.array([1.0, 0.0])

# Наблюдатель полного порядка (deadbeat)
L = np.array([[1.0/2.4], [2.0]])  # deadbeat наблюдатель

# Подбор начального состояния внутренней модели w(0) для минимизации ошибки (первые 3 c)
def simulate_with_w0(w0):
    x_loc = x.copy()
    xhat_loc = np.zeros(2)
    w_loc = w0.copy()
    err_acc = 0.0
    steps_3s = int(np.ceil(3.0 / T))
    for k in range(steps_3s):
        r = A_g * np.sin(omega * k * T)
        y = float(C @ x_loc)
        e = r - y
        u = float(-Kx @ xhat_loc - Kw @ w_loc)
        x_loc = Ad @ x_loc + Bd.flatten() * u
        w_loc = Aosc @ w_loc + Bosc.flatten() * e
        innovation = y - float(C @ xhat_loc)
        xhat_loc = Ad @ xhat_loc + Bd.flatten() * u + L.flatten() * innovation
        err_acc += e * e
    return err_acc

# Грубый поиск по сетке для w(0)
w_best = np.zeros(2)
best_cost = float('inf')
grid = np.linspace(-A_g, A_g, 9)
for w0_1 in grid:
    for w0_2 in grid:
        cost = simulate_with_w0(np.array([w0_1, w0_2]))
        if cost < best_cost:
            best_cost = cost
            w_best = np.array([w0_1, w0_2])

w = w_best.copy()

xhat = np.zeros(2)

# Истории
x1_hist, x2_hist = [], []
xhat1_hist, xhat2_hist = [], []
w1_hist, w2_hist = [], []
err_norm_hist = []
y_hist, r_hist, u_hist, e_hist = [], [], [], []

for k in range(N):
    r = A_g * np.sin(omega * k * T)
    y = float(C @ x)
    e = r - y

    u = float(-Kx @ xhat - Kw @ w)  # управление по оценке состояния

    # обновления объекта и внутренней модели
    x = Ad @ x + Bd.flatten() * u
    w = Aosc @ w + Bosc.flatten() * e

    # наблюдатель
    innovation = y - float(C @ xhat)
    xhat = Ad @ xhat + Bd.flatten() * u + L.flatten() * innovation

    # истории
    x1_hist.append(x[0]); x2_hist.append(x[1])
    xhat1_hist.append(xhat[0]); xhat2_hist.append(xhat[1])
    w1_hist.append(w[0]); w2_hist.append(w[1])
    err_norm_hist.append(np.linalg.norm(x - xhat))
    y_hist.append(y); r_hist.append(r); u_hist.append(u); e_hist.append(e)

# Графики 3x2
t = np.arange(N) * T
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(t, x1_hist, 'b-', label='x1')
plt.plot(t, xhat1_hist, 'r--', label='x1_hat')
plt.ylabel('x1'); plt.title('Состояние x1'); plt.grid(True); plt.legend()

plt.subplot(3, 2, 2)
plt.plot(t, x2_hist, 'b-', label='x2')
plt.plot(t, xhat2_hist, 'r--', label='x2_hat')
plt.ylabel('x2'); plt.title('Состояние x2'); plt.grid(True); plt.legend()

plt.subplot(3, 2, 3)
plt.plot(t, r_hist, 'r--', label='g(k)')
plt.plot(t, y_hist, 'b-', label='y(k)')
plt.ylabel('Амплитуда'); plt.title('Выход и задание'); plt.grid(True); plt.legend()

plt.subplot(3, 2, 4)
plt.plot(t, w1_hist, 'g-', label='w1')
plt.plot(t, w2_hist, 'm-', label='w2')
plt.ylabel('Сост. внутр. модели'); plt.title('Внутренняя модель'); plt.grid(True); plt.legend()

plt.subplot(3, 2, 5)
plt.plot(t, err_norm_hist, 'k-', label='||x - xhat||')
plt.xlabel('t, s'); plt.ylabel('Невязка'); plt.title('Невязка наблюдателя'); plt.grid(True); plt.legend()
plt.xlim(0, 3.0)

plt.subplot(3, 2, 6)
plt.plot(t, e_hist, 'r-', label='e(k) = g(k) - y(k)')
plt.xlabel('t, s'); plt.ylabel('Ошибка'); plt.title('Ошибка слежения'); plt.grid(True); plt.legend()
plt.xlim(0, 10.0)

plt.tight_layout()
out_img = IMG_DIR / 'fast_states.png'
plt.savefig(out_img, dpi=150)
plt.close()

print('График сохранен:', out_img)



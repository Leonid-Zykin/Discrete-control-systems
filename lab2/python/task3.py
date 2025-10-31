import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Пути
THIS_FILE = Path(__file__).resolve()
LAB_DIR = THIS_FILE.parent.parent
IMG_DIR = LAB_DIR / 'images' / 'task3'
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

# Внутренняя модель гармоники (резонатор) - как в задании 2
theta = omega * T
Aosc = np.array([[0.0, 1.0],
                 [-1.0, 2.0 * np.cos(theta)]])
Bosc = np.array([[0.0],
                 [1.0]])

# Расширенная система z = [x; w]
A_t = np.block([
    [Ad,               np.zeros((Ad.shape[0], 2))],
    [-Bosc @ C,        Aosc]
])
B_t = np.vstack([Bd, np.zeros((2, 1))])

# Синтезируем следящий регулятор точно как в задании 2
n = A_t.shape[0]
z_desired = np.zeros(n)
coeffs = np.poly(z_desired)
pA = np.zeros_like(A_t)
for i in range(n + 1):
    power = n - i
    if power == n:
        pA = pA + np.linalg.matrix_power(A_t, n)
    elif power >= 0:
        pA = pA + coeffs[i] * np.linalg.matrix_power(A_t, power)

W = np.hstack([B_t, A_t @ B_t, A_t @ A_t @ B_t, A_t @ A_t @ A_t @ B_t])
enT = np.zeros((1, n)); enT[0, -1] = 1.0
K_full = enT @ np.linalg.inv(W) @ pA

# Разделение на Kx и Kw
Kx = K_full[0, :2]
Kw = K_full[0, 2:]

print(f"Следящий регулятор: Kx={Kx}, Kw={Kw}")

# Наблюдатель полного порядка для объекта (deadbeat за 2 такта)
# Требуем (Ad - L C) с характеристическим многочленом λ^2 ⇒ след = 0, det = 0
# Для Ad = [[1,0],[2.4,1]] и C = [0 1] получаем L = [1/2.4, 2]^T
L = np.array([[1.0/2.4], [2.0]])

print(f"Наблюдатель: L={L.flatten()}")

# Моделирование замкнутой системы с наблюдателем
# Используем тот же регулятор из задания 2, но x заменяем на xhat
N = 200
x = np.array([1.0, 0.0])  # начальное состояние объекта
xhat = np.zeros(2)         # оценка состояния объекта
w = np.zeros(2)            # состояние внутренней модели

# История переменных
x1_hist, x2_hist = [], []
xhat1_hist, xhat2_hist = [], []
w1_hist, w2_hist = [], []
err_norm_hist = []
y_hist = []
u_hist = []
e_hist = []
r_hist = []

for k in range(N):
    # Гармоническое задание
    r = A_g * np.sin(omega * k * T)
    
    # Измерение выхода
    y = float(C @ x)
    
    # Ошибка слежения
    e = r - y
    
    # Обновление внутренней модели
    w = Aosc @ w + Bosc.flatten() * e
    
    # Управление с наблюдателем: u = -Kx*xhat - Kw*w
    u = float(-Kx @ xhat - Kw @ w)
    
    # Обновление объекта
    x = Ad @ x + Bd.flatten() * u
    
    # Обновление наблюдателя
    innovation = y - float(C @ xhat)
    xhat = Ad @ xhat + Bd.flatten() * u + L.flatten() * innovation
    
    # Сохранение истории
    x1_hist.append(x[0])
    x2_hist.append(x[1])
    xhat1_hist.append(xhat[0])
    xhat2_hist.append(xhat[1])
    w1_hist.append(w[0])
    w2_hist.append(w[1])
    err_norm_hist.append(np.linalg.norm(x - xhat))
    y_hist.append(y)
    u_hist.append(u)
    e_hist.append(e)
    r_hist.append(r)

# Построение графиков
t = np.arange(N) * T
plt.figure(figsize=(12, 10))

# График 1: Состояние объекта и наблюдателя
plt.subplot(3, 2, 1)
plt.plot(t, x1_hist, 'b-', label='x₁ (объект)')
plt.plot(t, xhat1_hist, 'r--', label='x̂₁ (наблюдатель)')
plt.ylabel('x₁')
plt.grid(True)
plt.legend()
plt.title('Переменная состояния x₁')

plt.subplot(3, 2, 2)
plt.plot(t, x2_hist, 'b-', label='x₂ (объект)')
plt.plot(t, xhat2_hist, 'r--', label='x̂₂ (наблюдатель)')
plt.ylabel('x₂')
plt.grid(True)
plt.legend()
plt.title('Переменная состояния x₂')

# График 2: Выход и задание
plt.subplot(3, 2, 3)
plt.plot(t, r_hist, 'r--', label='g(k)')
plt.plot(t, y_hist, 'b-', label='y(k)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()
plt.title('Выход и задание')

# График 3: Состояние внутренней модели
plt.subplot(3, 2, 4)
plt.plot(t, w1_hist, 'g-', label='w₁')
plt.plot(t, w2_hist, 'm-', label='w₂')
plt.ylabel('Состояние')
plt.grid(True)
plt.legend()
plt.title('Состояние внутренней модели')

# График 4: Невязка наблюдателя
plt.subplot(3, 2, 5)
plt.plot(t, err_norm_hist, 'm-', label='||x - x̂||')
plt.xlabel('t, s')
plt.ylabel('Невязка')
plt.grid(True)
plt.legend()
plt.title('Невязка наблюдателя')
# Подчеркнём deadbeat-сходимость за 2 такта
plt.xlim(0, 2*T)
plt.axvline(2*T, color='k', linestyle='--', alpha=0.5, label='t = 2T')

# График 5: Ошибка слежения
plt.subplot(3, 2, 6)
plt.plot(t, e_hist, 'r-', label='e(k) = g(k) - y(k)')
plt.xlabel('t, s')
plt.ylabel('Ошибка')
plt.grid(True)
plt.legend()
plt.title('Ошибка слежения')

plt.tight_layout()
plt.savefig(IMG_DIR / 'observer_states.png', dpi=150)
plt.close()

print('График сохранен:', IMG_DIR / 'observer_states.png')
print(f'Максимальная ошибка слежения: {max(abs(e) for e in e_hist):.6f}')
print(f'Ошибка в конце: {abs(e_hist[-1]):.6f}')
print(f'Максимальная невязка наблюдателя: {max(err_norm_hist):.6f}')
print(f'Невязка в конце: {err_norm_hist[-1]:.6f}')

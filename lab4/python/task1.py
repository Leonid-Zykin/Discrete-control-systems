import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# Вариант 8
a = 2.4  # запаздывание
b = 8.5  # постоянная времени
T = 1.0  # период дискретизации

# Пути
THIS = Path(__file__).resolve()
LAB = THIS.parent.parent
IMG = LAB / 'images' / 'task1'
IMG.mkdir(parents=True, exist_ok=True)

# Аппроксимация запаздывания Паде(1): e^{-a s} ≈ (1 - a s/2)/(1 + a s/2)
num_delay = np.array([1.0, -a/2])
den_delay = np.array([1.0,  a/2])
# Объект 1/(1 + b s)
plant = signal.TransferFunction([1.0], [b, 1.0])
# Полная непрерывная модель с запаздыванием
Gc = signal.TransferFunction(np.polymul(plant.num, num_delay), np.polymul(plant.den, den_delay))

# Дискретизация (ZOH)
Gd = signal.cont2discrete((Gc.num, Gc.den), T, method='zoh')
Bz = np.squeeze(Gd[0])
Az = np.squeeze(Gd[1])


def simulate_closed(K: float, N: int = 200):
    # Замкнутый с П-регулятором: U = K (R - Y)
    b = Bz.copy()
    a = Az.copy()
    # Нормировка
    b = b / a[0]
    a = a / a[0]
    # Полиномы замкнутой системы
    a_cl = np.polyadd(a, K * b)
    b_cl = K * b
    r = np.ones(N)
    y = signal.lfilter(b_cl, a_cl, r)
    return y


def overshoot(y):
    r = 1.0
    return max(0.0, (np.max(y) - r) / r)


def settle_time(y, tol=0.02):
    r = 1.0
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - r) > tol:
            return (i + 1) * T
    return 0.0


# Поиск K с апериодикой и минимальным временем установления
Ks = np.linspace(0.1, 20.0, 300)
bestK, bestTs = Ks[0], 1e9
for K in Ks:
    y = simulate_closed(K)
    if not np.isfinite(y).all():
        continue
    if overshoot(y) <= 1e-3:
        ts = settle_time(y)
        if ts < bestTs:
            bestTs, bestK = ts, K

# Финальный прогон
y = simulate_closed(bestK, N=300)
t = np.arange(len(y)) * T

plt.figure(figsize=(8, 4))
plt.plot(t, np.ones_like(t), 'k--', lw=0.8, label='r=1')
plt.plot(t, y, label=f'y, K={bestK:.2f}')
plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
plt.tight_layout()
plt.savefig(IMG / 'step_ap.png', dpi=150)
plt.close()

print(f'Aperiodic regulator gain K={bestK:.3f}, plot saved to', IMG / 'step_ap.png')

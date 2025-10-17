import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# Вариант 8
T = 1.0
b = 8.5

a = 2.4  # задержка (Паде(1))
num_delay = np.array([1.0, -a/2])
den_delay = np.array([1.0,  a/2])
plant = signal.TransferFunction([1.0], [b, 1.0])
Gc = signal.TransferFunction(np.polymul(plant.num, num_delay), np.polymul(plant.den, den_delay))

# Дискретизация ZOH для передаточной функции -> (numd, dend, dt)
Bz, Az, _ = signal.cont2discrete((Gc.num, Gc.den), T, method='zoh')
Bz = np.squeeze(Bz); Az = np.squeeze(Az)

# Целевая дискретная модель 1-го порядка (Даллин) с параметром alpha
Tm = b
alpha = np.exp(-T / Tm)

Gz_num = np.poly1d(Bz)
Gz_den = np.poly1d(Az)


def simulate(K, q, N=200):
    Cz_num = np.poly1d([K, -K*q])
    Cz_den = np.poly1d([1])
    open_num = np.polymul(Gz_num, Cz_num)
    open_den = np.polymul(Gz_den, Cz_den)
    cl_num = open_num
    cl_den = np.polyadd(open_den, open_num)
    b = cl_num.c
    a = cl_den.c
    b = b / a[0]; a = a / a[0]
    r = np.ones(N)
    y = signal.lfilter(b, a, r)
    return y

K_grid = np.linspace(0.1, 10.0, 60)
q_grid = np.linspace(0.0, 1.0, 50)
best = None
best_params = (1.0, 0.0)
for K in K_grid:
    for q in q_grid:
        y = simulate(K, q, N=300)
        if not np.isfinite(y).all():
            continue
        os = max(0.0, (y.max()-1.0))
        ts = 0
        for i in range(len(y)-1, -1, -1):
            if abs(y[i]-1.0) > 0.02:
                ts = i+1
                break
        score = 5*os + ts
        if best is None or score < best:
            best = score
            best_params = (K, q)

K_opt, q_opt = best_params
print(f'Dahlin: K={K_opt:.3f}, q={q_opt:.3f}')

y = simulate(K_opt, q_opt, N=350)
t = np.arange(len(y)) * T

THIS = Path(__file__).resolve()
IMG = THIS.parent.parent / 'images' / 'task2'
IMG.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8,4))
plt.plot(t, np.ones_like(t), 'k--', lw=0.8, label='r=1')
plt.plot(t, y, label=f'y, K={K_opt:.2f}, q={q_opt:.2f}')
plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
plt.tight_layout()
plt.savefig(IMG / 'step_dahlin.png', dpi=150)
plt.close()
print('Saved:', IMG / 'step_dahlin.png')

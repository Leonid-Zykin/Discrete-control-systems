import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Вариант 8
T1 = 0.9
T2 = 1.05

# Пути
THIS = Path(__file__).resolve()
LAB = THIS.parent.parent
IMG1 = LAB / 'images' / 'task1'
IMG1.mkdir(parents=True, exist_ok=True)

# Непрерывная модель объекта: G(s) = 1/(T1 s + 1) * 1/(T2 s + 1)
# В пространства состояний (каноника каскада)
A_c = np.array([[-1.0/T1, 0.0],
                [ 1.0/T2, -1.0/T2]])
B_c = np.array([[1.0/T1],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])


def simulate(Ts: float, q0: float = 1.0, N: int = 300, mode: str = 'set', seed: int = 0):
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')

    # Простая дискретная ПИ-подобная регуляция через фильтр первого порядка:
    # u_k = q0 * (r_k - y_k)
    # Можно позже заменить на рассчитанные параметры при необходимости
    x = np.zeros(2)
    y_hist, u_hist, r_hist, d_hist = [], [], [], []
    rng = np.random.default_rng(seed)
    for k in range(N):
        if mode == 'set':
            r = 1.0
            d = 0.0
        elif mode == 'dist_step':
            r = 0.0
            d = 0.5 if k >= 20 else 0.0
        elif mode == 'noise':
            r = 0.0
            d = 0.2 * rng.standard_normal()
        else:
            r = 0.0; d = 0.0

        y = float(C @ x)
        e = r - y
        u = q0 * e - d
        x = (Ad @ x) + (Bd.flatten() * u)

        y_hist.append(y); u_hist.append(u); r_hist.append(r); d_hist.append(d)
    t = np.arange(N) * Ts
    return t, np.array(r_hist), np.array(y_hist), np.array(u_hist), np.array(d_hist)


def plot_and_save(Ts: float, mode: str, fname: str, q0: float = 1.0):
    t, r, y, u, d = simulate(Ts, q0=q0, mode=mode)
    plt.figure(figsize=(8,4))
    if mode == 'set':
        plt.plot(t, r, label='r')
    if mode != 'set':
        plt.plot(t, d, label='disturbance')
    plt.plot(t, y, label='y')
    plt.xlabel('t, s'); plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.savefig(IMG1 / fname, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Периоды по заданию
    Ts12 = T1/2
    Ts14 = T1/4
    # Набросочный выбор q0, далее можно будет откорректировать
    q0 = 1.0

    # T = T1/2
    plot_and_save(Ts12, 'set', 'step_set_T12.png', q0)
    plot_and_save(Ts12, 'dist_step', 'step_dist_T12.png', q0)
    plot_and_save(Ts12, 'noise', 'noise_T12.png', q0)

    # T = T1/4
    plot_and_save(Ts14, 'set', 'step_set_T14.png', q0)
    plot_and_save(Ts14, 'dist_step', 'step_dist_T14.png', q0)
    plot_and_save(Ts14, 'noise', 'noise_T14.png', q0)

    print('Сохранены графики task1 для T=T1/2 и T1/4')

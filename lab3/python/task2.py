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
IMG2 = LAB / 'images' / 'task2'
IMG2.mkdir(parents=True, exist_ok=True)

# Непрерывная модель объекта
A_c = np.array([[-1.0/T1, 0.0],
                [ 1.0/T2, -1.0/T2]])
B_c = np.array([[1.0/T1],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])


def get_controller_params(Ts: float):
    """Расчёт по формуле методички: Wc(z)=q0 (z-d1)(z-d2)/(z(z-1))"""
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')
    d1 = np.exp(-Ts / T1)
    d2 = np.exp(-Ts / T2)
    num_ctrl = np.array([1.0, -(d1 + d2), d1 * d2])
    den_ctrl = np.array([1.0, -1.0, 0.0])
    return num_ctrl, den_ctrl, Ad, Bd, Cd, Dd


def simulate(Ts: float, q0: float, N: int = 400, mode: str = 'set'):
    """Моделирование системы с регулятором"""
    num_ctrl, den_ctrl, Ad, Bd, Cd, Dd = get_controller_params(Ts)
    
    if len(num_ctrl) > 0 and num_ctrl[0] != 0:
        num_ctrl = num_ctrl / num_ctrl[0]
    if len(den_ctrl) > 0 and den_ctrl[0] != 0:
        den_ctrl = den_ctrl / den_ctrl[0]
    
    n_ctrl = max(len(num_ctrl), len(den_ctrl))
    e_buf = np.zeros(n_ctrl)
    u_ctrl_buf = np.zeros(n_ctrl)
    
    x = np.zeros(Ad.shape[0])
    y_hist = []
    
    for k in range(N):
        if mode == 'set':
            r = 1.0
            d = 0.0
        elif mode == 'dist_step':
            r = 0.0
            d = 0.5 if k >= 20 else 0.0
        else:
            r = 0.0
            d = 0.0
        
        y = float(Cd @ x)
        
        e = r - y
        
        # Регулятор
        u = 0.0
        for i in range(len(num_ctrl)):
            if i < len(e_buf):
                u += q0 * num_ctrl[i] * e_buf[i]
        for i in range(1, len(den_ctrl)):
            if i < len(u_ctrl_buf):
                u -= den_ctrl[i] * u_ctrl_buf[i]
        
        e_buf = np.roll(e_buf, 1)
        e_buf[0] = e
        u_ctrl_buf = np.roll(u_ctrl_buf, 1)
        u_ctrl_buf[0] = u
        
        u_obj = u + d
        x = Ad @ x + Bd.flatten() * u_obj
        
        y_hist.append(y)
    
    t = np.arange(N) * Ts
    return t, np.array(y_hist)


def step_quality(y: np.ndarray, r: float = 1.0, Ts: float = 1.0):
    overshoot = max(0.0, (y.max() - r) / max(r, 1e-9))
    tol = 0.02 * r if r != 0 else 0.02
    settled_idx = len(y) - 1
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - r) > tol:
            break
        settled_idx = i
    t_s = settled_idx * Ts
    ise = np.mean((y - r) ** 2)
    return overshoot, t_s, ise


def tune_q0(Ts: float, q_min: float = 0.1, q_max: float = 5.0, nq: int = 60):
    best = None
    q_grid = np.linspace(q_min, q_max, nq)
    for q in q_grid:
        t, y = simulate(Ts, q, mode='set')
        os, t_s, ise = step_quality(y, Ts=Ts)
        if not np.isfinite(y).all():
            continue
        if 0.05 <= os <= 0.15:
            cand = (0, t_s, ise, q)
        else:
            cand = (1, ise, t_s, q)
        if best is None or cand < best:
            best = cand
    if best is None:
        return 1.0
    return float(best[-1])


if __name__ == '__main__':
    Ts12 = T1 / 2
    Ts14 = T1 / 4

    q12 = tune_q0(Ts12)
    q14 = tune_q0(Ts14)

    print(f"Подобранные q0: T=T1/2 => {q12:.4f}, T=T1/4 => {q14:.4f}")

    # Сохраняем q0 для обоих периодов
    (LAB / 'python' / 'q0_T12.txt').write_text(f"{q12:.6f}\n")
    (LAB / 'python' / 'q0_T14.txt').write_text(f"{q14:.6f}\n")

    # Сравнение периодов по пункту 4: при ступенчатом возмущении
    t12, y12 = simulate(Ts12, q12, mode='dist_step')
    t14, y14 = simulate(Ts14, q14, mode='dist_step')

    plt.figure(figsize=(8,4))
    d_signal = np.where(t12 >= 20 * Ts12, 0.5, 0.0)
    plt.step(t12, d_signal, where='post', color='k', lw=1.0, label='возмущение')
    plt.step(t12, y12, where='post', color='C0', lw=1.8, label=f'y, T=T1/2, q0={q12:.2f}')
    plt.step(t14, y14, where='post', color='C1', lw=1.8, label=f'y, T=T1/4, q0={q14:.2f}')
    plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.savefig(IMG2 / 'compare_T.png', dpi=150)
    plt.close()

    print('График сравнения сохранён')

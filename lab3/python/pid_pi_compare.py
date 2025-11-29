import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Вариант 8
T1 = 0.9
T2 = 1.05
Ts = T1 / 2  # используем T = T1/2 как в задании

# Пути
THIS = Path(__file__).resolve()
LAB = THIS.parent.parent
IMG = LAB / 'images' / 'task_extra'
IMG.mkdir(parents=True, exist_ok=True)

# Непрерывная модель объекта: два апериодических звена
A_c = np.array([[-1.0/T1, 0.0],
                [ 1.0/T2, -1.0/T2]])
B_c = np.array([[1.0/T1],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])

Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')


def simulate_controller(kind: str, Kp: float, Ki: float, Kd: float, N: int = 400):
    """Моделирование замкнутой системы с PI/PID-регулятором (скоростная форма)."""
    x = np.zeros(2)
    y_hist = []
    u_hist = []

    # Буферы ошибок для производной
    e_prev = 0.0
    e_prev2 = 0.0
    u_prev = 0.0

    for k in range(N):
        r = 1.0  # единичная ступень
        y = float(C @ x)
        e = r - y

        # Скоростные формулы
        du = Kp * (e - e_prev) + Ki * Ts * e
        if kind.upper() == 'PID':
            du += (Kd / Ts) * (e - 2.0 * e_prev + e_prev2)

        u = u_prev + du

        # Объект
        x = (Ad @ x) + (Bd.flatten() * u)

        # Сохранение
        y_hist.append(y)
        u_hist.append(u)

        # Сдвиги
        e_prev2 = e_prev
        e_prev = e
        u_prev = u

    t = np.arange(N) * Ts
    return t, np.array(y_hist)


if __name__ == '__main__':
    # Набор умеренных настроек (подобрано вручную для наглядности)
    # PI
    Kp_pi = 2.0
    Ki_pi = 0.6
    # PID
    Kp_pid = 2.0
    Ki_pid = 0.6
    Kd_pid = 0.15

    t_pi, y_pi = simulate_controller('PI', Kp_pi, Ki_pi, 0.0)
    t_pid, y_pid = simulate_controller('PID', Kp_pid, Ki_pid, Kd_pid)

    # Отдельно: PI
    plt.figure(figsize=(9, 4))
    plt.step(t_pi, np.ones_like(t_pi), where='post', color='k', lw=1.0, label='r=1')
    plt.step(t_pi, y_pi, where='post', color='C1', lw=1.8, label=f'PI: Kp={Kp_pi}, Ki={Ki_pi}')
    plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
    plt.tight_layout()
    out_pi = IMG / 'pi_step.png'
    plt.savefig(out_pi, dpi=150)
    plt.close()

    # Отдельно: PID
    plt.figure(figsize=(9, 4))
    plt.step(t_pid, np.ones_like(t_pid), where='post', color='k', lw=1.0, label='r=1')
    plt.step(t_pid, y_pid, where='post', color='C0', lw=1.8, label=f'PID: Kp={Kp_pid}, Ki={Ki_pid}, Kd={Kd_pid}')
    plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
    plt.tight_layout()
    out_pid = IMG / 'pid_step.png'
    plt.savefig(out_pid, dpi=150)
    plt.close()

    print('Графики сохранены:', out_pi, out_pid)



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
IMG = LAB / 'images' / 'task_extra'
IMG.mkdir(parents=True, exist_ok=True)

def get_discrete_model(Ts: float):
    A_c = np.array([[-1.0/T1, 0.0],
                    [ 1.0/T2, -1.0/T2]])
    B_c = np.array([[1.0/T1],
                    [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')
    return Ad, Bd.flatten(), Cd, float(Dd)

def simulate_pid_closed(Ts: float, Kp: float, Ki: float, Kd: float, N: int = 400):
    Ad, Bd, Cd, Dd = get_discrete_model(Ts)
    x = np.zeros(2)
    e_prev = 0.0
    e_prev2 = 0.0
    u_prev = 0.0

    y_hist, u_hist = [], []
    for k in range(N):
        r = 1.0
        y = float(Cd @ x)
        e = r - y

        du = Kp * (e - e_prev) + Ki * Ts * e + (Kd / Ts) * (e - 2.0 * e_prev + e_prev2)
        u = u_prev + du

        x = Ad @ x + Bd * u

        y_hist.append(y)
        u_hist.append(u)

        e_prev2 = e_prev
        e_prev = e
        u_prev = u

    t = np.arange(N) * Ts
    return t, np.array(y_hist), np.array(u_hist)

def simulate_pid_pure(Ts: float, Kp: float, Ki: float, Kd: float, N: int = 200, return_parts: bool = False):
    # Выход регулятора на вход e(k)=1 (единичный шаг)
    e_prev = 0.0
    e_prev2 = 0.0
    u_prev = 0.0
    u_hist = []
    uP_hist, uI_hist, uD_hist = [], [], []
    for k in range(N):
        e = 1.0
        dP = Kp * (e - e_prev)
        dI = Ki * Ts * e
        dD = (Kd / Ts) * (e - 2.0 * e_prev + e_prev2)
        du = dP + dI + dD
        u = u_prev + du
        u_hist.append(u)
        uP_hist.append(u_prev + dP)
        uI_hist.append(u_prev + dI)
        uD_hist.append(u_prev + dD)
        e_prev2 = e_prev
        e_prev = e
        u_prev = u
    t = np.arange(N) * Ts
    if return_parts:
        return t, np.array(u_hist), np.array(uP_hist), np.array(uI_hist), np.array(uD_hist)
    return t, np.array(u_hist)

def simulate_p_pure(Ts: float, Kp: float, N: int = 200):
    # u = Kp * e, e=1
    u = np.full(N, Kp)
    t = np.arange(N) * Ts
    return t, u

def simulate_pi_pure(Ts: float, Kp: float, Ki: float, N: int = 200):
    e_prev = 0.0
    u_prev = 0.0
    u_hist = []
    for k in range(N):
        e = 1.0
        du = Kp * (e - e_prev) + Ki * Ts * e
        u = u_prev + du
        u_hist.append(u)
        e_prev = e
        u_prev = u
    t = np.arange(N) * Ts
    return t, np.array(u_hist)

def simulate_i_pure(Ts: float, Ki: float, N: int = 200):
    # u(k) = u(k-1) + Ki*Ts*e(k), e=1
    u = 0.0
    u_hist = []
    for _ in range(N):
        u += Ki * Ts * 1.0
        u_hist.append(u)
    t = np.arange(N) * Ts
    return t, np.array(u_hist)

def simulate_d_pure(Ts: float, Kd: float, N: int = 200):
    # Классическое D без накопления: u(k) = (Kd/Ts)*(e(k)-e(k-1)), e=1
    # Даёт импульс в первый отсчёт и далее ноль (на ступени).
    u_hist = []
    e_prev = 0.0
    for k in range(N):
        e = 1.0
        u = (Kd / Ts) * (e - e_prev)
        u_hist.append(u)
        e_prev = e
    t = np.arange(N) * Ts
    return t, np.array(u_hist)

if __name__ == '__main__':
    # Настройки (умеренные для иллюстрации)
    Kp = 2.0
    Ki = 0.6
    Kd = 0.15

    for Ts, tag in [(T1/2, 'T12'), (T1/4, 'T14')]:
        t, y, u = simulate_pid_closed(Ts, Kp, Ki, Kd)

        plt.figure(figsize=(9,5))
        plt.subplot(2,1,1)
        plt.step(t, np.ones_like(t), where='post', color='k', lw=1.0, label='r=1')
        plt.step(t, y, where='post', color='C0', lw=1.8, label='y')
        plt.ylabel('y'); plt.grid(True); plt.legend()

        plt.subplot(2,1,2)
        plt.step(t, u, where='post', color='C1', lw=1.4, label='u')
        plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.legend()
        plt.tight_layout()
        out = IMG / f'pid_closed_{tag}.png'
        plt.savefig(out, dpi=150)
        plt.close()

        # Отдельный рисунок: только реакция y на единичный шаг (для T=T1/2 — основной запрос)
        if tag == 'T12':
            plt.figure(figsize=(8,3.6))
            plt.step(t, np.ones_like(t), where='post', color='k', lw=1.0, label='r=1')
            plt.step(t, y, where='post', color='C0', lw=1.8, label='y (PID)')
            plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend(); plt.tight_layout()
            out_resp = IMG / 'pid_response_T12.png'
            plt.savefig(out_resp, dpi=150)
            plt.close()

        # Чистый PID
        t_p, u_p = simulate_pid_pure(Ts, Kp, Ki, Kd)
        plt.figure(figsize=(9,4))
        plt.step(t_p, u_p, where='post', color='C2', lw=1.6, label='u_PID на шаг e')
        plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.legend(); plt.tight_layout()
        out_p = IMG / f'pid_pure_{tag}.png'
        plt.savefig(out_p, dpi=150)
        plt.close()
        print('Saved:', out, out_p)

    # Сравнение P, PI, PID (чистые регуляторы) для T=T1/2
    Ts = T1/2
    # Настройка для наглядности различий: делаем зум и усиливаем D только для чистого PID-графика
    Kd_vis = 0.6
    N_vis = 120
    tP, uP = simulate_p_pure(Ts, Kp, N=N_vis)
    tI, uI = simulate_i_pure(Ts, Ki, N=N_vis)
    tD, uD = simulate_d_pure(Ts, Kd_vis, N=N_vis)
    tPI, uPI = simulate_pi_pure(Ts, Kp, Ki, N=N_vis)
    tPID, uPID, _, _, _ = simulate_pid_pure(Ts, Kp, Ki, Kd_vis, N=N_vis, return_parts=True)
    # Отдельные изображения для P и PI
    plt.figure(figsize=(9,4))
    plt.step(tP, uP, where='post', color='C3', lw=1.6)
    plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.tight_layout()
    plt.xlim(0.0, 2.5)
    out_p_only = IMG / 'pure_P_T12.png'
    plt.savefig(out_p_only, dpi=150)
    plt.close()

    plt.figure(figsize=(9,4))
    plt.step(tPI, uPI, where='post', color='C1', lw=1.6)
    plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.tight_layout()
    plt.xlim(0.0, 2.5)
    out_pi_only = IMG / 'pure_PI_T12.png'
    plt.savefig(out_pi_only, dpi=150)
    plt.close()

    # Отдельные изображения I и D
    plt.figure(figsize=(9,4))
    plt.step(tI, uI, where='post', color='C0', lw=1.6)
    plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.tight_layout(); plt.xlim(0.0, 2.5)
    out_i_only = IMG / 'pure_I_T12.png'
    plt.savefig(out_i_only, dpi=150)
    plt.close()

    plt.figure(figsize=(9,4))
    plt.step(tD, uD, where='post', color='C4', lw=1.6)
    plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.tight_layout(); plt.xlim(0.0, 2.5)
    out_d_only = IMG / 'pure_D_T12.png'
    plt.savefig(out_d_only, dpi=150)
    plt.close()

    # Сводная картинка (на всякий случай оставим тоже)
    plt.figure(figsize=(9,4))
    plt.step(tP, uP, where='post', color='C3', lw=1.6, label='P')
    plt.step(tPI, uPI, where='post', color='C1', lw=1.6, label='PI')
    plt.step(tPID, uPID, where='post', color='C2', lw=1.6, label='PID (усиленный D)')
    plt.xlabel('t, s'); plt.ylabel('u'); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.xlim(0.0, 2.5)
    out_cmp = IMG / 'pure_P_PI_PID_T12.png'
    plt.savefig(out_cmp, dpi=150)
    plt.close()
    print('Saved compare:', out_cmp, 'and singles:', out_p_only, out_pi_only, out_i_only, out_d_only)



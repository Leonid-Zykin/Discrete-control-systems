#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

# Параметры варианта 8
T = 0.2  # период дискретизации, с
K_CO = 3.4  # усиление непрерывной части

# Директория для сохранения рисунков
ROOT = Path(__file__).resolve().parent.parent  # lab1/
IMG_DIR = ROOT / "images" / "task1"
IMG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class StepResponseResult:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray


def simulate_closed_loop(K_fb: float, t_end: float = 10.0) -> StepResponseResult:
    """Моделирование замкнутой системы с Wc(s)=K_CO/s, ZOH, периодом T.
    Модель состояния берём как интегратор: x_{k+1} = x_k + T * (K_CO * u_k)
    Выход y_k = x_k. Управление u_k = r_k - K_fb * y_k (единичная обратная связь с коэффициентом).
    """
    n_steps = int(np.ceil(t_end / T)) + 1
    t = np.arange(n_steps) * T
    x = 0.0
    y_hist = np.zeros(n_steps)
    u_hist = np.zeros(n_steps)
    r = 1.0  # единичный скачок на входе

    for k in range(n_steps):
        y = x
        u = r - K_fb * y
        # интегратор с коэффициентом K_CO и ZOH (u постоянен в интервале)
        if k < n_steps - 1:  # не обновляем на последнем шаге
            x = x + T * (K_CO * u)
        y_hist[k] = y
        u_hist[k] = u

    return StepResponseResult(t=t, y=y_hist, u=u_hist)


def save_step_plot(res: StepResponseResult, title: str, fname: str) -> None:
    plt.figure(figsize=(8, 4))
    # Ступенчатая функция (ZOH) - более корректно для дискретных систем
    plt.step(res.t, res.y, where='post', label='y(k)', linewidth=2)
    plt.step(res.t, np.ones_like(res.t), where='post', label='r(k)=1', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('t, c')
    plt.ylabel('Амплитуда')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = IMG_DIR / f"{fname}.png"
    plt.savefig(out_path.as_posix(), dpi=160)
    plt.close()


def save_scheme_placeholder():
    """Простой плейсхолдер для схемы из методички."""
    plt.figure(figsize=(6, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, 'Схема из методички\n(ZOH, K_FB, K_CO, 1/s)',
             ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig((IMG_DIR / 'scheme_placeholder.png').as_posix(), dpi=160)
    plt.close()


def find_stability_boundaries() -> dict:
    """Поиск приблизительных границ устойчивости по модулю собственного числа.
    Для линейной дискретной системы первого порядка с интегратором получаем:
        x_{k+1} = (1 - T*K_CO*K_fb) * x_k + T*K_CO
    Собственное число a = 1 - T*K_CO*K_fb. Граница по модулю |a|=1.
    => нейтральная граница: K_fb = 0 (a=1) и K_fb = 2/(T*K_CO) (a=-1).
    Возвращаем характерные K_fb и описательные подписи.
    """
    a_to_K = lambda a: (1 - a) / (T * K_CO)
    K_neutral_plus = a_to_K(1.0)   # 0
    K_neutral_minus = a_to_K(-1.0) # 2/(T*K_CO)
    return {
        'neutral_pos': K_neutral_plus,
        'neutral_neg': K_neutral_minus,
    }


def main():
    save_scheme_placeholder()
    bounds = find_stability_boundaries()
    # Нейтральные границы
    for name, kfb in bounds.items():
        res = simulate_closed_loop(kfb, t_end=10.0)
        save_step_plot(res, f'Граница устойчивости: {name}, K_FB={kfb:.3f}',
                       f'step_boundary_{"neutral" if name=="neutral_pos" else "osc"}')

    # Без колебаний (апериодическое): выберем a=0.3 для более выраженного затухания
    a = 0.3
    K_no_osc = (1 - a) / (T * K_CO)
    res_no = simulate_closed_loop(K_no_osc, t_end=10.0)
    save_step_plot(res_no, f'Без колебаний, K_FB={K_no_osc:.3f}', 'step_no_osc')

    # Максимальная колебательность на грани: a≈-1 -> возьмём a=-0.9
    a = -0.9
    K_max_osc = (1 - a) / (T * K_CO)
    res_mo = simulate_closed_loop(K_max_osc, t_end=10.0)
    save_step_plot(res_mo, f'Макс. колебательность, K_FB={K_max_osc:.3f}', 'step_max_osc')

    # Критерий быстродействия: минимизация времени установления ~ выберем a=0.1
    a = 0.1
    K_fast = (1 - a) / (T * K_CO)
    res_fast = simulate_closed_loop(K_fast, t_end=10.0)
    save_step_plot(res_fast, f'Быстродействие, K_FB={K_fast:.3f}', 'step_fast')


if __name__ == '__main__':
    main()

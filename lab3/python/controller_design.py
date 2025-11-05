"""
Расчет параметров дискретного регулятора с компенсацией полюсов
Структура регулятора: W(z) = q0 * num(z) / (z^2 - z)
"""
import numpy as np
from scipy.signal import cont2discrete, tf2zpk

# Вариант 8
T1 = 0.9
T2 = 1.05


def design_controller(Ts: float):
    """
    Расчет параметров регулятора для периода дискретизации Ts
    
    Объект: G(s) = 1/((T1*s+1)*(T2*s+1))
    После дискретизации через ZOH: G(z) = B(z)/A(z)
    
    Структура регулятора: W(z) = q0 * num_controller(z) / (z^2 - z)
    где num_controller(z) компенсирует полюса объекта A(z)
    """
    # Непрерывная передаточная функция объекта
    # G(s) = 1 / ((T1*s+1)*(T2*s+1))
    num_s = np.array([1.0])
    den_s = np.array([T1*T2, T1+T2, 1.0])  # (T1*s+1)*(T2*s+1) = T1*T2*s^2 + (T1+T2)*s + 1
    
    # Дискретизация через ZOH
    num_z, den_z, _ = cont2discrete((num_s, den_s), Ts, method='zoh')
    num_z = np.squeeze(num_z)
    den_z = np.squeeze(den_z)
    
    # Нормализация (старший коэффициент знаменателя = 1)
    num_z = num_z / den_z[0]
    den_z = den_z / den_z[0]
    
    # Полюса объекта (корни знаменателя)
    poles_obj, _, _ = tf2zpk(num_z, den_z)
    
    # Структура регулятора: W(z) = q0 * num_controller(z) / (z^2 - z)
    # Для компенсации полюсов объекта: num_controller(z) должен содержать A(z)
    # Но (z^2 - z) уже есть в знаменателе регулятора, поэтому
    # num_controller(z) = A(z) / (z^2 - z) для компенсации
    
    # Упрощенный подход: компенсируем только устойчивые полюса объекта
    stable_poles = [p for p in poles_obj if abs(p) < 1.0]
    
    if len(stable_poles) > 0:
        # num_controller(z) компенсирует устойчивые полюса
        # Но так как знаменатель регулятора (z^2 - z), 
        # а объект имеет полюса из A(z), то num_controller = часть A(z), соответствующая полюсам
        num_controller = np.poly(stable_poles)
        num_controller = num_controller / num_controller[0] if len(num_controller) > 0 and num_controller[0] != 0 else np.array([1.0])
    else:
        num_controller = np.array([1.0])
    
    # Знаменатель регулятора: z^2 - z
    den_controller = np.array([1.0, -1.0, 0.0])  # z^2 - z
    
    return {
        'num_controller': num_controller,
        'den_controller': den_controller,
        'num_object': num_z,
        'den_object': den_z,
        'poles_object': poles_obj,
        'stable_poles': stable_poles
    }


if __name__ == '__main__':
    # Тестирование расчета регулятора
    Ts12 = T1 / 2
    Ts14 = T1 / 4
    
    print("Расчет регулятора для T = T1/2:")
    ctrl12 = design_controller(Ts12)
    print(f"  Передаточная функция объекта: num={ctrl12['num_object']}, den={ctrl12['den_object']}")
    print(f"  Полюса объекта: {ctrl12['poles_object']}")
    print(f"  Числитель регулятора: {ctrl12['num_controller']}")
    print(f"  Знаменатель регулятора: {ctrl12['den_controller']}")
    
    print("\nРасчет регулятора для T = T1/4:")
    ctrl14 = design_controller(Ts14)
    print(f"  Передаточная функция объекта: num={ctrl14['num_object']}, den={ctrl14['den_object']}")
    print(f"  Полюса объекта: {ctrl14['poles_object']}")
    print(f"  Числитель регулятора: {ctrl14['num_controller']}")
    print(f"  Знаменатель регулятора: {ctrl14['den_controller']}")

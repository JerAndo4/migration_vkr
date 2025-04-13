import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel
from hybrid_model import HybridModel

# Константы для моделирования
SIMULATION_TIME = 1000  # количество тактов моделирования
NUM_NODES = 4          # количество узлов
NUM_SERVICES = 20      # количество сервисов

def ensure_directories():
    """Создает необходимые директории для результатов и графиков"""
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def initialize_system():
    """
    Инициализация начального состояния системы
    
    Возвращает:
    -----------
    tuple : (node_loads, service_allocation, service_loads)
    """
    # Начальная загрузка узлов (равномерная, 50%)
    node_loads = np.ones(NUM_NODES) * 0.5
    
    # Начальное распределение сервисов (равномерное)
    service_allocation = np.zeros(NUM_SERVICES, dtype=int)
    for i in range(NUM_SERVICES):
        service_allocation[i] = i % NUM_NODES
    
    # Начальная загрузка, создаваемая сервисами (случайная в пределах 5-15%)
    service_loads = np.random.uniform(0.05, 0.15, NUM_SERVICES)
    
    return node_loads, service_allocation, service_loads

def generate_standard_scenario():
    """
    Генерирует стандартный сценарий с плавными колебаниями нагрузки
    
    Возвращает:
    -----------
    numpy.ndarray : Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    """
    # Инициализация матрицы загрузки
    loads = np.zeros((SIMULATION_TIME, NUM_NODES))
    
    # Начальная загрузка в центре желаемого диапазона [0.6, 0.9]
    base_load = np.ones(NUM_NODES) * 0.7
    
    # Генерация плавных колебаний в пределах [0.6, 0.9]
    for t in range(SIMULATION_TIME):
        # Синусоидальные колебания с различными периодами для каждого узла
        for i in range(NUM_NODES):
            period = 100 + i * 20  # разные периоды для разных узлов
            amplitude = 0.15  # амплитуда колебаний
            offset = i * np.pi / 4  # фазовый сдвиг
            
            # Базовая загрузка + колебания
            loads[t, i] = base_load[i] + amplitude * np.sin(2 * np.pi * t / period + offset)
            
        # Добавление небольшого случайного шума
        loads[t] += np.random.normal(0, 0.02, NUM_NODES)
        
        # Ограничение значений в пределах [0.6, 0.9]
        loads[t] = np.clip(loads[t], 0.6, 0.9)
    
    return loads

def generate_critical_scenario():
    """
    Генерирует критический сценарий с резким повышением нагрузки на один из узлов
    
    Возвращает:
    -----------
    numpy.ndarray : Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    """
    # Инициализация матрицы загрузки
    loads = np.zeros((SIMULATION_TIME, NUM_NODES))
    
    # Начальная загрузка (50%)
    base_load = np.ones(NUM_NODES) * 0.5
    
    # Генерация плавных колебаний со случайными критическими скачками
    for t in range(SIMULATION_TIME):
        # Базовая загрузка с небольшими колебаниями
        for i in range(NUM_NODES):
            period = 100 + i * 20
            amplitude = 0.05
            offset = i * np.pi / 4
            loads[t, i] = base_load[i] + amplitude * np.sin(2 * np.pi * t / period + offset)
        
        # Добавление небольшого случайного шума
        loads[t] += np.random.normal(0, 0.02, NUM_NODES)
        
        # Создание критических ситуаций
        if 200 <= t < 250:  # Критическая ситуация на узле 0
            loads[t, 0] = 0.9 + np.random.normal(0, 0.02)
        elif 400 <= t < 450:  # Критическая ситуация на узле 1
            loads[t, 1] = 0.95 + np.random.normal(0, 0.02)
        elif 600 <= t < 650:  # Критическая ситуация на узле 2
            loads[t, 2] = 0.9 + np.random.normal(0, 0.02)
        elif 800 <= t < 850:  # Критическая ситуация на узле 3
            loads[t, 3] = 0.95 + np.random.normal(0, 0.02)
        
        # Ограничение значений в пределах [0.4, 0.98]
        loads[t] = np.clip(loads[t], 0.4, 0.98)

    return loads

def generate_dynamic_scenario():
    """
    Генерирует сценарий с частыми непредсказуемыми изменениями нагрузки
    
    Возвращает:
    -----------
    numpy.ndarray : Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    """
    # Инициализация матрицы загрузки
    loads = np.zeros((SIMULATION_TIME, NUM_NODES))
    
    # Начальная загрузка (50%)
    loads[0] = np.ones(NUM_NODES) * 0.5
    
    # Генерация случайных изменений нагрузки с определенной инерцией
    for t in range(1, SIMULATION_TIME):
        # Инерция - новая загрузка зависит от предыдущей
        inertia = 0.7
        random_change = np.random.normal(0, 0.1, NUM_NODES)
        
        loads[t] = inertia * loads[t-1] + (1 - inertia) * random_change
        
        # Добавление случайных скачков
        if np.random.random() < 0.05:  # 5% вероятность скачка
            node = np.random.randint(0, NUM_NODES)
            direction = 1 if np.random.random() < 0.5 else -1
            loads[t, node] += direction * np.random.uniform(0.2, 0.4)
        
        # Ограничение значений в пределах [0.3, 0.85]
        loads[t] = np.clip(loads[t], 0.3, 0.85)

    # Добавляем несколько пиков нагрузки для стимуляции миграций
    for i in range(5):
        start_time = np.random.randint(50, SIMULATION_TIME - 100)
        duration = np.random.randint(10, 30)
        node = np.random.randint(0, NUM_NODES)
        
        # Увеличиваем нагрузку для вызова миграций
        loads[start_time:start_time+duration, node] = np.clip(
            loads[start_time:start_time+duration, node] + 0.3, 0, 0.95
        )
    
    return loads

def generate_mixed_scenario():
    """
    Генерирует смешанный сценарий, комбинирующий различные типы нагрузки
    
    Возвращает:
    -----------
    numpy.ndarray : Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    """
    # Инициализация матрицы загрузки
    loads = np.zeros((SIMULATION_TIME, NUM_NODES))
    
    # Начальная загрузка (50%)
    loads[0] = np.ones(NUM_NODES) * 0.5
    
    # Разделение симуляции на 5 сегментов с разными типами нагрузки
    segment_size = SIMULATION_TIME // 5
    
    # Стандартный сценарий (плавные колебания) - сегмент 1
    standard_loads = generate_standard_scenario()
    loads[:segment_size] = standard_loads[:segment_size]
    
    # Критический сценарий - сегмент 2
    critical_loads = generate_critical_scenario()
    loads[segment_size:2*segment_size] = critical_loads[segment_size:2*segment_size]
    
    # Динамический сценарий - сегмент 3
    dynamic_loads = generate_dynamic_scenario()
    loads[2*segment_size:3*segment_size] = dynamic_loads[2*segment_size:3*segment_size]
    
    # Для сегментов 4-5 используем комбинированный подход
    for t in range(3*segment_size, SIMULATION_TIME):
        # Добавление элементов всех сценариев
        node = t % NUM_NODES
        
        if t % 50 < 10:  # Периодические критические скачки
            loads[t, node] = 0.9 + np.random.normal(0, 0.03)
        elif t % 20 < 5:  # Частые колебания
            loads[t, node] = 0.7 + np.random.normal(0, 0.05)
        else:  # Стандартная нагрузка
            loads[t, node] = 0.5 + 0.1 * np.sin(t / 30) + np.random.normal(0, 0.02)
        
        # Для остальных узлов - стандартная нагрузка с колебаниями
        for i in range(NUM_NODES):
            if i != node:
                loads[t, i] = 0.5 + 0.1 * np.sin(t / 30 + i * np.pi / 2) + np.random.normal(0, 0.02)
        
        # Ограничение значений в пределах [0.3, 0.95]
        loads[t] = np.clip(loads[t], 0.3, 0.95)

    return loads

def generate_limited_resources_scenario():
    """
    Генерирует сценарий с ограниченными ресурсами, когда общая нагрузка постоянно высокая
    
    Возвращает:
    -----------
    numpy.ndarray : Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    """
    # Инициализация матрицы загрузки
    loads = np.zeros((SIMULATION_TIME, NUM_NODES))
    
    # Начальная высокая загрузка (70-80%)
    loads[0] = np.random.uniform(0.7, 0.8, NUM_NODES)
    
    # Генерация нагрузки с высоким базовым уровнем
    for t in range(1, SIMULATION_TIME):
        # Базовый высокий уровень нагрузки
        for i in range(NUM_NODES):
            period = 80 + i * 15
            amplitude = 0.1
            offset = i * np.pi / 3
            
            # Базовая загрузка + колебания
            base_value = 0.75 + amplitude * np.sin(2 * np.pi * t / period + offset)
            
            # Инерция от предыдущего значения
            inertia = 0.6
            loads[t, i] = inertia * loads[t-1, i] + (1 - inertia) * base_value
        
        # Добавление случайных флуктуаций
        loads[t] += np.random.normal(0, 0.05, NUM_NODES)
        
        # Ограничение значений в пределах [0.65, 0.95]
        loads[t] = np.clip(loads[t], 0.65, 0.95)
    
    return loads

def run_scenario(loads, model_name, model, scenario_name):
    """
    Запускает один сценарий с заданной моделью миграции и обеспечивает
    корректный сбор всех метрик производительности
    
    Параметры:
    ----------
    loads : numpy.ndarray
        Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    model_name : str
        Название модели для записи в результаты
    model : object
        Экземпляр модели миграции
    scenario_name : str
        Название сценария
    
    Возвращает:
    -----------
    pandas.DataFrame : Результаты симуляции
    """
    start_time = time.time()
    
    # Инициализация системы
    node_loads, service_allocation, service_loads = initialize_system()
    model.initialize_system(node_loads, service_allocation, service_loads)
    
    # Инициализация результатов - все массивы должны быть инициализированы здесь
    results = {
        'time': [], 
        'model': [], 
        'latency': [], 
        'jitter': [], 
        'energy_consumption': [],
        'migration_performed': [], 
        'migration_success': [], 
        'migration_mode': [],
        'cpu_utilization': [], 
        'ram_utilization': [], 
        'overloaded_nodes': [],
        'predicted_overloads': [], 
        'migrations': []
    }
    
    # Добавим cumulative_energy только если планируем его использовать
    if 'cumulative_energy' not in results:
        results['cumulative_energy'] = []
    
    # Счетчики для отслеживания миграций
    total_migrations = 0
    reactive_migrations = 0
    proactive_migrations = 0
    cumulative_energy = 0
    
    # Запуск симуляции
    simulation_steps = min(SIMULATION_TIME, len(loads))
    for t in range(simulation_steps):
        # Установка текущей нагрузки
        current_loads = loads[t].copy()
        
        # Обработка одного шага
        step_metrics = model.process_step(current_loads)
        
        # Подсчет миграций
        if step_metrics.get('migration_performed', False) and step_metrics.get('migration_success', False):
            total_migrations += 1
            if step_metrics.get('migration_mode') == 'reactive':
                reactive_migrations += 1
            elif step_metrics.get('migration_mode') == 'proactive':
                proactive_migrations += 1
        
        # Получение текущих метрик из модели
        current_energy = 0
        current_latency = 0
        current_jitter = 0
        
        # Безопасное получение метрик
        if hasattr(model, 'metrics'):
            if 'energy_consumption' in model.metrics and len(model.metrics['energy_consumption']) > 0:
                current_energy = model.metrics['energy_consumption'][-1]
            if 'latency' in model.metrics and len(model.metrics['latency']) > 0:
                current_latency = model.metrics['latency'][-1]
            if 'jitter' in model.metrics and len(model.metrics['jitter']) > 0:
                current_jitter = model.metrics['jitter'][-1]
        
        # Обновление кумулятивного энергопотребления
        cumulative_energy += current_energy
        
        # Запись всех результатов для текущего шага - все массивы должны пополняться здесь
        results['time'].append(t)
        results['model'].append(model_name)
        results['latency'].append(current_latency)
        results['jitter'].append(current_jitter)
        results['energy_consumption'].append(current_energy)
        results['cumulative_energy'].append(cumulative_energy)
        results['migration_performed'].append(step_metrics.get('migration_performed', False))
        results['migration_success'].append(step_metrics.get('migration_success', False))
        results['migration_mode'].append(step_metrics.get('migration_mode', None))
        results['cpu_utilization'].append(np.mean(current_loads))
        results['ram_utilization'].append(np.mean(current_loads) * 0.8)  # Условная метрика
        
        # Подсчет перегруженных узлов и прогнозов
        overloaded_count = np.sum(current_loads > 0.75)
        predicted_overload_count = 0
        
        if model_name in ['qlearning', 'hybrid']:
            # Для моделей с прогнозированием
            predicted_load, _ = model.q_learning_model.predict_future_load() if hasattr(model, 'q_learning_model') else (current_loads, 0)
            predicted_overload_count = np.sum(predicted_load > 0.65)
        
        results['overloaded_nodes'].append(overloaded_count)
        results['predicted_overloads'].append(predicted_overload_count)
        results['migrations'].append(total_migrations)
    
    # Проверка, что все массивы имеют одинаковую длину
    lengths = {key: len(value) for key, value in results.items() if isinstance(value, list)}
    if len(set(lengths.values())) > 1:
        print(f"ОШИБКА: Разная длина массивов в результатах: {lengths}")
        # Обрежем все массивы до минимальной длины
        min_length = min(lengths.values())
        for key in results:
            if isinstance(results[key], list) and len(results[key]) > min_length:
                results[key] = results[key][:min_length]
    
    # Создание DataFrame из результатов
    df = pd.DataFrame(results)
    
    # Добавляем скользящее среднее для сглаживания графиков
    window_size = 5
    if len(df) >= window_size:
        df['smooth_latency'] = df['latency'].rolling(window=window_size, min_periods=1).mean()
        df['smooth_jitter'] = df['jitter'].rolling(window=window_size, min_periods=1).mean()
        df['smooth_energy'] = df['energy_consumption'].rolling(window=window_size, min_periods=1).mean()
    
    elapsed_time = time.time() - start_time
    print(f"Симуляция для {model_name} в сценарии {scenario_name} завершена за {elapsed_time:.2f} секунд")
    print(f"  Всего миграций: {total_migrations}")
    if model_name == 'hybrid':
        print(f"  Реактивных миграций: {reactive_migrations}")
        print(f"  Проактивных миграций: {proactive_migrations}")
    print(f"  Общее энергопотребление: {cumulative_energy:.2f} кВт·ч")

    return df
    
    # Создание DataFrame из результатов
    df = pd.DataFrame(results)
    
    # Добавляем скользящее среднее для сглаживания графиков
    window_size = 5
    df['smooth_latency'] = df['latency'].rolling(window=window_size, min_periods=1).mean()
    df['smooth_jitter'] = df['jitter'].rolling(window=window_size, min_periods=1).mean()
    df['smooth_energy'] = df['energy_consumption'].rolling(window=window_size, min_periods=1).mean()
    
    # Расчет энергоэффективности (отношение задержки к энергопотреблению)
    # Более низкое значение означает более высокую эффективность
    df['energy_efficiency'] = df['latency'] / (df['energy_consumption'] + 0.001)  # Избегаем деления на ноль
    
    # Запись статистики в логи
    elapsed_time = time.time() - start_time
    # if DEBUG:
    #     print(f"Симуляция для {model_name} в сценарии {scenario_name} завершена за {elapsed_time:.2f} секунд")
    #     print(f"  Всего миграций: {total_migrations}")
    #     if model_name == 'hybrid':
    #         print(f"  Реактивных миграций: {reactive_migrations}")
    #         print(f"  Проактивных миграций: {proactive_migrations}")
    #     print(f"  Общее энергопотребление: {cumulative_energy:.2f} кВт·ч")
    #     print(f"  Средняя задержка: {df['latency'].mean():.2f} мс")
    #     print(f"  Средний джиттер: {df['jitter'].mean():.2f} мс")

    return df

def run_scenarios():
    """
    Запускает все сценарии для всех моделей и сохраняет результаты
    
    Возвращает:
    -----------
    dict : Словарь с результатами для каждого сценария
    """
    # Проверяем и создаем директории
    ensure_directories()
    
    # Генерация сценариев
    scenarios = {
        'standard': generate_standard_scenario(),
        'critical': generate_critical_scenario(),
        'dynamic': generate_dynamic_scenario(),
        'mixed': generate_mixed_scenario(),
        'limited_resources': generate_limited_resources_scenario()
    }
    
    # Инициализация моделей
    threshold_model = ThresholdModel(NUM_NODES, NUM_SERVICES)
    q_learning_model = QLearningModel(NUM_NODES, NUM_SERVICES)
    hybrid_model = HybridModel(NUM_NODES, NUM_SERVICES)
    
    # Запуск симуляций и сбор результатов
    results = {}
    
    for scenario_name, loads in scenarios.items():
        
        # Запуск сценария для каждой модели
        threshold_results = run_scenario(loads, 'threshold', threshold_model, scenario_name)
        q_learning_results = run_scenario(loads, 'qlearning', q_learning_model, scenario_name)
        hybrid_results = run_scenario(loads, 'hybrid', hybrid_model, scenario_name)
        
        # Объединение результатов
        scenario_results = pd.concat([threshold_results, q_learning_results, hybrid_results])
        
        # Сохранение результатов для каждой модели отдельно
        threshold_results.to_csv(f'results/{scenario_name}_threshold.csv', index=False)
        q_learning_results.to_csv(f'results/{scenario_name}_qlearning.csv', index=False)
        hybrid_results.to_csv(f'results/{scenario_name}_hybrid.csv', index=False)
        
        # Сохранение общих результатов
        scenario_results.to_csv(f'results/{scenario_name}_results.csv', index=False)
        
        # Сбор сводных метрик
        summary = []
        for model_name, model in [('threshold', threshold_model), ('qlearning', q_learning_model), ('hybrid', hybrid_model)]:
            model_metrics = model.get_metrics()
            
            # Базовые метрики для всех моделей
            model_summary = {
                'scenario': scenario_name,
                'model': model_name,
                'avg_migrations': model_metrics['migrations_count'] / max(1, SIMULATION_TIME),
                'total_migrations': model_metrics['migrations_count'],
                'avg_latency': model_metrics['avg_latency'],
                'avg_jitter': model_metrics['avg_jitter'],
                'avg_cpu': np.mean(scenario_results['cpu_utilization']),
                'avg_ram': np.mean(scenario_results['ram_utilization']),
                'avg_energy': model_metrics['avg_energy_consumption'],
                'max_latency': max(threshold_results['latency'].max(), q_learning_results['latency'].max(), hybrid_results['latency'].max()),
                'max_jitter': max(threshold_results['jitter'].max(), q_learning_results['jitter'].max(), hybrid_results['jitter'].max())
            }
            
            # Добавление специфичных метрик для гибридной модели
            if model_name == 'hybrid':
                model_summary['reactive_migrations'] = model_metrics['reactive_migrations']
                model_summary['proactive_migrations'] = model_metrics['proactive_migrations']
            
            # Добавляем в список
            summary.append(model_summary)
        
        # Сохранение сводных метрик в формате, удобном для plot_results.py
        summary_df = pd.DataFrame(summary)
        
        # Преобразуем из широкого формата в длинный для совместимости
        summary_long = []
        for _, row in summary_df.iterrows():
            model = row['model']
            scenario = row['scenario']
            
            for metric, value in row.items():
                if metric not in ['model', 'scenario']:
                    summary_long.append({
                        'scenario': scenario,
                        'model': model,
                        'metric': metric,
                        'value': value
                    })
        
        # Сохранение сводных метрик
        pd.DataFrame(summary_long).to_csv(f'results/{scenario_name}_summary.csv', index=False)
        
        # Добавление результатов в общий словарь
        results[scenario_name] = scenario_results
        
    
    return results

def main():
    """
    Основная функция для запуска всего процесса симуляции
    """
    # Запуск сценариев
    results = run_scenarios()
    
    return results

if __name__ == "__main__":
    main()
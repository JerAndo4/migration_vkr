import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel
from hybrid_model import HybridModel

# Константы для моделирования
SIMULATION_TIME = 1000  # количество тактов моделирования
NUM_NODES = 4          # количество узлов
NUM_SERVICES = 20      # количество сервисов

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
    
    # Начальная загрузка (50%)
    base_load = np.ones(NUM_NODES) * 0.5
    
    # Генерация плавных колебаний в пределах 50-70%
    for t in range(SIMULATION_TIME):
        # Синусоидальные колебания с различными периодами для каждого узла
        for i in range(NUM_NODES):
            period = 100 + i * 20  # разные периоды для разных узлов
            amplitude = 0.1  # амплитуда колебаний
            offset = i * np.pi / 4  # фазовый сдвиг
            
            # Базовая загрузка + колебания
            loads[t, i] = base_load[i] + amplitude * np.sin(2 * np.pi * t / period + offset)
            
        # Добавление небольшого случайного шума
        loads[t] += np.random.normal(0, 0.02, NUM_NODES)
        
        # Ограничение значений в пределах [0.5, 0.7]
        loads[t] = np.clip(loads[t], 0.5, 0.7)
    
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
    
    return loads

def generate_periodic_scenario():
    """
    Генерирует сценарий с периодическими пиками нагрузки
    
    Возвращает:
    -----------
    numpy.ndarray : Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    """
    # Инициализация матрицы загрузки
    loads = np.zeros((SIMULATION_TIME, NUM_NODES))
    
    # Начальная загрузка (40%)
    base_load = np.ones(NUM_NODES) * 0.4
    
    # Периоды пиков нагрузки для каждого узла
    periods = [120, 150, 180, 200]
    
    # Генерация периодических пиков нагрузки
    for t in range(SIMULATION_TIME):
        for i in range(NUM_NODES):
            # Базовая загрузка
            loads[t, i] = base_load[i]
            
            # Периодические пики
            phase = (t % periods[i]) / periods[i]
            
            # Создание пика в определенной фазе (нарастание и спад)
            if phase < 0.2:  # Нарастание
                loads[t, i] += 0.3 * (phase / 0.2)
            elif phase < 0.3:  # Плато
                loads[t, i] += 0.3
            elif phase < 0.5:  # Спад
                loads[t, i] += 0.3 * (1 - (phase - 0.3) / 0.2)
        
        # Добавление небольшого случайного шума
        loads[t] += np.random.normal(0, 0.02, NUM_NODES)
        
        # Ограничение значений в пределах [0.35, 0.85]
        loads[t] = np.clip(loads[t], 0.35, 0.85)
    
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
    
    # Периодический сценарий - сегмент 4
    periodic_loads = generate_periodic_scenario()
    loads[3*segment_size:4*segment_size] = periodic_loads[3*segment_size:4*segment_size]
    
    # Комбинированный сегмент - сегмент 5
    for t in range(4*segment_size, SIMULATION_TIME):
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

def run_scenario(loads, model_name, model):
    """
    Запускает один сценарий с заданной моделью миграции
    
    Параметры:
    ----------
    loads : numpy.ndarray
        Матрица загрузки узлов на каждом шаге симуляции [SIMULATION_TIME, NUM_NODES]
    model_name : str
        Название модели для записи в результаты
    model : object
        Экземпляр модели миграции
    
    Возвращает:
    -----------
    pandas.DataFrame : Результаты симуляции
    """
    # Инициализация системы
    node_loads, service_allocation, service_loads = initialize_system()
    model.initialize_system(node_loads, service_allocation, service_loads)
    
    # Инициализация результатов
    results = {
        'step': [], 'model': [], 'latency': [], 'jitter': [], 
        'energy_consumption': [], 'migration_performed': [], 'migration_success': []
    }
    
    # Запуск симуляции
    for t in range(SIMULATION_TIME):
        # Установка текущей нагрузки
        current_loads = loads[t].copy()
        
        # Обработка одного шага
        step_metrics = model.process_step(current_loads)
        
        # Запись результатов
        results['step'].append(t)
        results['model'].append(model_name)
        results['latency'].append(step_metrics['latency'])
        results['jitter'].append(step_metrics['jitter'])
        results['energy_consumption'].append(0 if t >= len(model.metrics['energy_consumption']) else 
                                           model.metrics['energy_consumption'][-1] if model.metrics['energy_consumption'] else 0)
        results['migration_performed'].append(step_metrics['migration_performed'])
        results['migration_success'].append(step_metrics['migration_success'])
    
    # Создание DataFrame из результатов
    return pd.DataFrame(results)

def run_scenarios():
    """
    Запускает все сценарии для всех моделей и сохраняет результаты
    
    Возвращает:
    -----------
    dict : Словарь с результатами для каждого сценария
    """
    # Генерация сценариев
    scenarios = {
        'standard': generate_standard_scenario(),
        'critical': generate_critical_scenario(),
        'dynamic': generate_dynamic_scenario(),
        'periodic': generate_periodic_scenario(),
        'mixed': generate_mixed_scenario()
    }
    
    # Инициализация моделей
    threshold_model = ThresholdModel(NUM_NODES, NUM_SERVICES)
    q_learning_model = QLearningModel(NUM_NODES, NUM_SERVICES)
    hybrid_model = HybridModel(NUM_NODES, NUM_SERVICES)
    
    # Запуск симуляций и сбор результатов
    results = {}
    
    for scenario_name, loads in scenarios.items():
        # Запуск сценария для каждой модели
        threshold_results = run_scenario(loads, 'threshold', threshold_model)
        q_learning_results = run_scenario(loads, 'q_learning', q_learning_model)
        hybrid_results = run_scenario(loads, 'hybrid', hybrid_model)
        
        # Объединение результатов
        scenario_results = pd.concat([threshold_results, q_learning_results, hybrid_results])
        
        # Сохранение результатов
        scenario_results.to_csv(f'{scenario_name}_results.csv', index=False)
        
        # Сбор сводных метрик
        summary = {}
        for model_name in ['threshold', 'q_learning', 'hybrid']:
            model_data = scenario_results[scenario_results['model'] == model_name]
            
            summary[model_name] = {
                'avg_latency': model_data['latency'].mean(),
                'avg_jitter': model_data['jitter'].mean(),
                'avg_energy': model_data['energy_consumption'].mean(),
                'migrations_count': model_data['migration_performed'].sum(),
                'success_rate': (model_data['migration_success'].sum() / 
                                max(1, model_data['migration_performed'].sum()))
            }
        
        # Сохранение сводных метрик
        pd.DataFrame(summary).T.to_csv(f'{scenario_name}_summary.csv')
        
        # Добавление результатов в общий словарь
        results[scenario_name] = scenario_results
    
    return results

def plot_results(scenario_name, results_df):
    """
    Визуализирует результаты симуляции
    
    Параметры:
    ----------
    scenario_name : str
        Название сценария
    results_df : pandas.DataFrame
        Результаты симуляции
    """
    # Создание графиков
    plt.figure(figsize=(15, 12))
    plt.suptitle(f'Результаты симуляции для сценария "{scenario_name}"', fontsize=16)
    
    # Подготовка данных по моделям
    models = ['threshold', 'q_learning', 'hybrid']
    colors = ['r', 'g', 'b']
    
    # График задержки
    plt.subplot(2, 2, 1)
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        plt.plot(model_data['step'], model_data['latency'], color=colors[i], alpha=0.7, label=model)
    plt.title('Задержка')
    plt.xlabel('Шаг симуляции')
    plt.ylabel('Задержка (мс)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График джиттера
    plt.subplot(2, 2, 2)
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        plt.plot(model_data['step'], model_data['jitter'], color=colors[i], alpha=0.7, label=model)
    plt.title('Джиттер')
    plt.xlabel('Шаг симуляции')
    plt.ylabel('Джиттер (мс)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График энергопотребления
    plt.subplot(2, 2, 3)
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        plt.plot(model_data['step'], model_data['energy_consumption'], color=colors[i], alpha=0.7, label=model)
    plt.title('Энергопотребление')
    plt.xlabel('Шаг симуляции')
    plt.ylabel('Энергопотребление (условные единицы)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График миграций
    plt.subplot(2, 2, 4)
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        # Накопительная сумма миграций
        migrations = model_data['migration_performed'].cumsum()
        plt.plot(model_data['step'], migrations, color=colors[i], alpha=0.7, label=model)
    plt.title('Количество миграций (накопительно)')
    plt.xlabel('Шаг симуляции')
    plt.ylabel('Количество миграций')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сохранение графиков
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{scenario_name}_plots.png', dpi=150)
    plt.close()

def plot_summary(scenarios):
    """
    Визуализирует сводные результаты по всем сценариям
    
    Параметры:
    ----------
    scenarios : list
        Список названий сценариев
    """
    # Загрузка сводных данных
    summary_data = {}
    
    for scenario in scenarios:
        summary_df = pd.read_csv(f'{scenario}_summary.csv', index_col=0)
        summary_data[scenario] = summary_df
    
    # Подготовка данных для графиков
    metrics = ['avg_latency', 'avg_jitter', 'avg_energy', 'migrations_count', 'success_rate']
    metric_names = ['Средняя задержка (мс)', 'Средний джиттер (мс)', 'Среднее энергопотребление', 
                    'Количество миграций', 'Успешность миграций (%)']
    
    models = ['threshold', 'q_learning', 'hybrid']
    model_colors = {'threshold': 'r', 'q_learning': 'g', 'hybrid': 'b'}
    
    # Создание графиков для каждой метрики
    plt.figure(figsize=(15, 18))
    plt.suptitle('Сводные результаты по всем сценариям', fontsize=16)
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(3, 2, i+1)
        
        # Подготовка данных для графика
        x = np.arange(len(scenarios))
        width = 0.25
        
        for j, model in enumerate(models):
            values = [summary_data[scenario].loc[model, metric] for scenario in scenarios]
            
            # Для процентов - умножаем на 100
            if metric == 'success_rate':
                values = [val * 100 for val in values]
                
            plt.bar(x + (j - 1) * width, values, width, label=model, color=model_colors[model], alpha=0.7)
        
        plt.title(metric_name)
        plt.xticks(x, scenarios)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Сохранение графика
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('summary_plots.png', dpi=150)
    plt.close()
    
    # Дополнительный график для гибридной модели - соотношение реактивных и проактивных миграций
    # Для этого нам нужно получить эти данные из файлов результатов
    hybrid_migrations = {}
    
    for scenario in scenarios:
        results_df = pd.read_csv(f'{scenario}_results.csv')
        hybrid_data = results_df[results_df['model'] == 'hybrid']
        
        # Подсчет миграций по режимам
        reactive_count = 0
        proactive_count = 0
        
        for i, row in hybrid_data.iterrows():
            if row['migration_performed']:
                if 'migration_mode' in row and row['migration_mode'] == 'reactive':
                    reactive_count += 1
                else:
                    proactive_count += 1
        
        hybrid_migrations[scenario] = {
            'reactive': reactive_count,
            'proactive': proactive_count,
            'total': reactive_count + proactive_count
        }
    
    # Вывод статистики по гибридной модели для каждого сценария
    print("Статистика миграций по сценариям:")
    for scenario in scenarios:
        stats = hybrid_migrations[scenario]
        print(f"Сценарий: {scenario}")
        print(f"  Пороговая модель: {summary_data[scenario].loc['threshold', 'migrations_count']}")
        print(f"  Q-Learning модель: {summary_data[scenario].loc['q_learning', 'migrations_count']}")
        print(f"  Гибридная модель (всего): {stats['total']}")
        print(f"    - Реактивные: {stats['reactive']}")
        print(f"    - Проактивные: {stats['proactive']}")
    
    # Создание графика соотношения реактивных и проактивных миграций
    plt.figure(figsize=(12, 7))
    plt.title('Соотношение реактивных и проактивных миграций в гибридной модели', fontsize=14)
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    reactive_values = [hybrid_migrations[scenario]['reactive'] for scenario in scenarios]
    proactive_values = [hybrid_migrations[scenario]['proactive'] for scenario in scenarios]
    
    plt.bar(x - width/2, reactive_values, width, label='Реактивные', color='r', alpha=0.7)
    plt.bar(x + width/2, proactive_values, width, label='Проактивные', color='g', alpha=0.7)
    
    plt.xlabel('Сценарии')
    plt.ylabel('Количество миграций')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_migrations_ratio.png', dpi=150)
    plt.close()

def main():
    """
    Основная функция для запуска всего процесса симуляции и визуализации
    """
    # Запуск сценариев
    results = run_scenarios()
    
    # Визуализация результатов по каждому сценарию
    for scenario_name, scenario_results in results.items():
        plot_results(scenario_name, scenario_results)
    
    # Визуализация сводных результатов
    plot_summary(list(results.keys()))
    
    print("Симуляция и визуализация завершены. Результаты сохранены в файлы.")

if __name__ == "__main__":
    main()
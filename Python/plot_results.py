import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib as mpl
from tabulate import tabulate
import sys

# Настройка вывода для отладки
DEBUG = True

# Настройка шрифтов для корректного отображения русских символов
mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# Настройка стиля графиков
plt.style.use('ggplot')
COLOR_THRESHOLD = 'steelblue'
COLOR_QLEARNING = 'seagreen'
COLOR_HYBRID = 'indianred'
COLOR_HYBRID_REACTIVE = 'indianred'
COLOR_HYBRID_PROACTIVE = 'darkorange'

def check_file_exists(filename):
    """
    Проверяет наличие файла и выводит информацию
    
    Параметры:
    ----------
    filename : str
        Путь к файлу
        
    Возвращает:
    ----------
    bool : True, если файл существует
    """
    if os.path.exists(filename):
        if DEBUG:
            print(f"Файл найден: {filename}")
        return True
    else:
        if DEBUG:
            print(f"Предупреждение: Файл не найден: {filename}")
        return False

def load_results(scenario_name, model_name):
    """
    Загрузка результатов симуляции с проверкой разных возможных путей
    
    Параметры:
    ----------
    scenario_name : str
        Название сценария
    model_name : str
        Название модели
        
    Возвращает:
    ----------
    pandas.DataFrame или None : данные или None, если файл не найден
    """
    # Список возможных путей к файлам
    possible_paths = [
        f'results/{scenario_name}_{model_name}.csv',   # Ожидаемый формат в results/
        f'{scenario_name}_{model_name}.csv',           # Возможно в корневой директории
        f'results/{scenario_name}_results.csv'         # Общий файл с результатами в results/
    ]
    
    # Пробуем загрузить из любого доступного пути
    for path in possible_paths:
        if check_file_exists(path):
            try:
                df = pd.read_csv(path)
                
                # Если это общий файл результатов, фильтруем по модели
                if 'model' in df.columns and model_name != 'all':
                    df = df[df['model'] == model_name]
                    
                if DEBUG:
                    print(f"Загружен файл {path}, {len(df)} строк")
                return df
            except Exception as e:
                if DEBUG:
                    print(f"Ошибка при загрузке файла {path}: {e}")
                continue
    
    if DEBUG:
        print(f"Предупреждение: Файл не найден: results/{scenario_name}_{model_name}.csv")
    return None

def find_summary_files():
    """
    Поиск файлов с сводными результатами
    
    Возвращает:
    ----------
    list : список путей к файлам
    """
    # Список возможных путей для поиска
    possible_paths = [
        'results/*_summary.csv',
        '*_summary.csv'
    ]
    
    # Собираем все найденные файлы
    summary_files = []
    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            summary_files.extend(files)
            if DEBUG:
                print(f"Найдены файлы по шаблону {pattern}: {len(files)}")
    
    if not summary_files:
        if DEBUG:
            print("Файлы с сводными результатами не найдены.")
    else:
        if DEBUG:
            print(f"Всего найдено файлов с сводными результатами: {len(summary_files)}")
            for file in summary_files[:5]:  # Выводим первые 5 для примера
                print(f"  - {file}")
            if len(summary_files) > 5:
                print(f"  ... и еще {len(summary_files) - 5}")
    
    return summary_files

def plot_all_models_migrations():
    """
    График 1: Сравнение количества миграций для всех моделей по всем сценариям
    с разделением на реактивные и проактивные для гибридной модели
    """
    if DEBUG:
        print("Генерация графика сравнения миграций для всех моделей...")
    
    plt.figure(figsize=(15, 8))
    
    # Список всех summary файлов
    summary_files = find_summary_files()
    
    if not summary_files:
        print("Файлы с результатами не найдены")
        return
    
    # Загружаем и объединяем все данные
    all_data = []
    for file in summary_files:
        try:
            data = pd.read_csv(file)
            all_data.append(data)
            if DEBUG:
                print(f"Загружены данные из {file}, {len(data)} строк")
        except Exception as e:
            print(f"Ошибка при загрузке файла {file}: {e}")
    
    if not all_data:
        print("Не удалось загрузить данные из файлов")
        return
    
    summary_df = pd.concat(all_data, ignore_index=True)
    
    # Определяем русские названия сценариев
    scenario_display_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический',
        'limited_resources': 'Ограниченные\nресурсы'
    }
    
    # Отфильтровываем только нужные сценарии
    main_scenarios = ['critical', 'standard', 'mixed', 'dynamic', 'limited_resources']
    
    # Фильтруем данные по миграциям
    migration_data = summary_df[summary_df['metric'] == 'total_migrations']
    migration_data = migration_data[migration_data['scenario'].isin(main_scenarios)]
    
    if migration_data.empty:
        print("Данные о миграциях не найдены в summary файлах")
        return
    
    # Создаем DataFrame для визуализации
    plot_data = {}
    for scenario in main_scenarios:
        plot_data[scenario] = {}
        for model in ['threshold', 'qlearning', 'hybrid']:
            model_data = migration_data[(migration_data['scenario'] == scenario) & 
                                      (migration_data['model'] == model)]
            if not model_data.empty:
                plot_data[scenario][model] = model_data['value'].values[0]
            else:
                if DEBUG:
                    print(f"Отсутствуют данные для сценария {scenario}, модель {model}")
                plot_data[scenario][model] = 0
    
    # Получаем данные о разделении миграций для гибридной модели
    hybrid_reactive = {}
    hybrid_proactive = {}
    
    # Ищем данные о разделении миграций в summary файлах
    for scenario in main_scenarios:
        # Ищем данные о реактивных миграциях
        reactive_data = summary_df[(summary_df['scenario'] == scenario) & 
                                (summary_df['model'] == 'hybrid') &
                                (summary_df['metric'] == 'reactive_migrations')]
        if not reactive_data.empty:
            hybrid_reactive[scenario] = reactive_data['value'].values[0]
        else:
            # Оцениваем как 50% от общего количества, если нет точных данных
            total = plot_data[scenario].get('hybrid', 0)
            hybrid_reactive[scenario] = total * 0.5
            if DEBUG:
                print(f"Данные о реактивных миграциях для сценария {scenario} не найдены. Используется оценка.")
        
        # Ищем данные о проактивных миграциях
        proactive_data = summary_df[(summary_df['scenario'] == scenario) & 
                                (summary_df['model'] == 'hybrid') &
                                (summary_df['metric'] == 'proactive_migrations')]
        if not proactive_data.empty:
            hybrid_proactive[scenario] = proactive_data['value'].values[0]
        else:
            # Оцениваем как 50% от общего количества, если нет точных данных
            total = plot_data[scenario].get('hybrid', 0)
            hybrid_proactive[scenario] = total * 0.5
            if DEBUG:
                print(f"Данные о проактивных миграциях для сценария {scenario} не найдены. Используется оценка.")
    
    # Ширина баров
    width = 0.25
    
    # Позиции для каждой группы баров
    x = np.arange(len(main_scenarios))
    
    # Получаем значения для отображения
    scenario_names = [scenario_display_names.get(s, s) for s in main_scenarios]
    threshold_values = [plot_data[scenario].get('threshold', 0) for scenario in main_scenarios]
    qlearning_values = [plot_data[scenario].get('qlearning', 0) for scenario in main_scenarios]
    
    # Получаем значения для гибридной модели
    reactive_values = [hybrid_reactive.get(scenario, 0) for scenario in main_scenarios]
    proactive_values = [hybrid_proactive.get(scenario, 0) for scenario in main_scenarios]
    
    # Выводим данные для отладки
    if DEBUG:
        print("\nДанные о миграциях:")
        for i, scenario in enumerate(main_scenarios):
            print(f"Сценарий: {scenario}")
            print(f"  Пороговая модель: {threshold_values[i]}")
            print(f"  Q-Learning модель: {qlearning_values[i]}")
            print(f"  Гибридная модель (всего): {reactive_values[i] + proactive_values[i]}")
            print(f"    - Реактивные: {reactive_values[i]}")
            print(f"    - Проактивные: {proactive_values[i]}")
    
    # Создаем бары (только если есть данные)
    if any(threshold_values) or any(qlearning_values) or any(reactive_values) or any(proactive_values):
        plt.bar(x - width, threshold_values, width, label='Пороговая модель', color=COLOR_THRESHOLD)
        plt.bar(x, qlearning_values, width, label='Q-Learning модель', color=COLOR_QLEARNING)
        
        # Создаем составной бар для гибридной модели
        plt.bar(x + width, reactive_values, width, label='Гибридная (реактивные)', color=COLOR_HYBRID_REACTIVE)
        plt.bar(x + width, proactive_values, width, bottom=reactive_values, 
                label='Гибридная (проактивные)', color=COLOR_HYBRID_PROACTIVE)
        
        # Настраиваем график
        plt.xlabel('Сценарий', fontsize=12)
        plt.ylabel('Количество миграций', fontsize=12)
        plt.title('Сравнение количества миграций по всем сценариям и моделям', fontsize=14, fontweight='bold')
        plt.xticks(x, scenario_names)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Сохраняем график
        plt.tight_layout()
        plt.savefig('plots/all_models_migrations.png', dpi=300, bbox_inches='tight')
        if DEBUG:
            print("График сохранен: plots/all_models_migrations.png")
    else:
        print("Нет данных для построения графика миграций")
    
    plt.close()

def plot_model_performance_per_scenario(scenario_name):
    """
    График 2: Для каждого сценария создает один график с задержкой и джиттером для каждой модели
    
    Параметры:
    ----------
    scenario_name : str
        Название сценария
    """
    if DEBUG:
        print(f"Обработка сценария: {scenario_name}")
    
    # Загружаем данные для всех моделей
    threshold_df = load_results(scenario_name, 'threshold')
    qlearning_df = load_results(scenario_name, 'qlearning')
    hybrid_df = load_results(scenario_name, 'hybrid')
    
    if threshold_df is None or qlearning_df is None or hybrid_df is None:
        print(f"Недостаточно данных для построения графиков для сценария {scenario_name}")
        return
    
    # Русские названия моделей и сценариев
    model_display_names = {
        'threshold': 'Пороговая модель',
        'qlearning': 'Q-Learning модель',
        'hybrid': 'Гибридная модель'
    }
    
    scenario_display_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический',
        'limited_resources': 'Ограниченные ресурсы'
    }
    
    scenario_title = scenario_display_names.get(scenario_name, scenario_name)
    
    # График с задержкой и джиттером для каждой модели
    plt.figure(figsize=(15, 10))
    
    # Подграфик для пороговой модели
    plt.subplot(3, 1, 1)
    plt.plot(threshold_df['time'], threshold_df['latency'], label='Задержка', color='blue', linewidth=2)
    plt.plot(threshold_df['time'], threshold_df['jitter'], label='Джиттер', color='red', linewidth=2)
    plt.title(f'Задержка и джиттер: {model_display_names["threshold"]}', fontsize=12)
    plt.ylabel('мс', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Подграфик для Q-Learning модели
    plt.subplot(3, 1, 2)
    plt.plot(qlearning_df['time'], qlearning_df['latency'], label='Задержка', color='blue', linewidth=2)
    plt.plot(qlearning_df['time'], qlearning_df['jitter'], label='Джиттер', color='red', linewidth=2)
    plt.title(f'Задержка и джиттер: {model_display_names["qlearning"]}', fontsize=12)
    plt.ylabel('мс', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Подграфик для гибридной модели
    plt.subplot(3, 1, 3)
    plt.plot(hybrid_df['time'], hybrid_df['latency'], label='Задержка', color='blue', linewidth=2)
    plt.plot(hybrid_df['time'], hybrid_df['jitter'], label='Джиттер', color='red', linewidth=2)
    plt.title(f'Задержка и джиттер: {model_display_names["hybrid"]}', fontsize=12)
    plt.xlabel('Время (такты)', fontsize=10)
    plt.ylabel('мс', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Общий заголовок
    plt.suptitle(f'Метрики задержки и джиттера для сценария: {scenario_title}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'plots/{scenario_name}_latency_jitter.png', dpi=300, bbox_inches='tight')
    
    if DEBUG:
        print(f"График сохранен: plots/{scenario_name}_latency_jitter.png")
    
    plt.close()
    
def plot_energy_comparison(scenario_name):
    """
    График сравнения энергопотребления между моделями
    
    Параметры:
    ----------
    scenario_name : str
        Название сценария
    """
    # Загружаем данные для всех моделей
    threshold_df = load_results(scenario_name, 'threshold')
    qlearning_df = load_results(scenario_name, 'qlearning')
    hybrid_df = load_results(scenario_name, 'hybrid')
    
    if threshold_df is None or qlearning_df is None or hybrid_df is None:
        print(f"Недостаточно данных для построения графиков энергопотребления для сценария {scenario_name}")
        return
    
    # Русские названия сценариев
    scenario_display_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический',
        'limited_resources': 'Ограниченные ресурсы'
    }
    
    scenario_title = scenario_display_names.get(scenario_name, scenario_name)
    
    # График энергопотребления для всех моделей
    plt.figure(figsize=(12, 6))
    
    # Убедимся, что все модели видны, используем разные цвета и стили линий
    plt.plot(threshold_df['time'], threshold_df['energy_consumption'], 
             label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2, alpha=0.9)
    plt.plot(qlearning_df['time'], qlearning_df['energy_consumption'], 
             label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2, alpha=0.9)
    plt.plot(hybrid_df['time'], hybrid_df['energy_consumption'], 
             label='Гибридная модель', color=COLOR_HYBRID, linewidth=2, alpha=0.9)
    
    plt.title(f'Энергопотребление для сценария: {scenario_title}', fontsize=14, fontweight='bold')
    plt.xlabel('Время (такты)', fontsize=12)
    plt.ylabel('Энергопотребление (Вт)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'plots/{scenario_name}_energy.png', dpi=300, bbox_inches='tight')
    
    if DEBUG:
        print(f"График сохранен: plots/{scenario_name}_energy.png")
    
    plt.close()

# def plot_resource_utilization(scenario_name):
    """
    Создает графики утилизации CPU и RAM для всех моделей
    
    Параметры:
    ----------
    scenario_name : str
        Название сценария
    """
    # Загружаем данные для всех моделей
    threshold_df = load_results(scenario_name, 'threshold')
    qlearning_df = load_results(scenario_name, 'qlearning')
    hybrid_df = load_results(scenario_name, 'hybrid')
    
    if threshold_df is None or qlearning_df is None or hybrid_df is None:
        print(f"Недостаточно данных для построения графиков утилизации ресурсов для сценария {scenario_name}")
        return
    
    # Убедимся, что нужные колонки присутствуют
    required_columns = ['time', 'cpu_utilization', 'ram_utilization']
    if not all(col in threshold_df.columns for col in required_columns):
        print(f"Отсутствуют необходимые колонки в данных для сценария {scenario_name}")
        return
    
    # Русские названия сценариев
    scenario_display_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический',
        'limited_resources': 'Ограниченные ресурсы'
    }
    
    scenario_title = scenario_display_names.get(scenario_name, scenario_name)
    
    # График утилизации ресурсов
    plt.figure(figsize=(12, 8))
    
    # График утилизации CPU
    plt.subplot(2, 1, 1)
    plt.plot(threshold_df['time'], threshold_df['cpu_utilization'], 
             label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2)
    plt.plot(qlearning_df['time'], qlearning_df['cpu_utilization'], 
             label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2)
    plt.plot(hybrid_df['time'], hybrid_df['cpu_utilization'], 
             label='Гибридная модель', color=COLOR_HYBRID, linewidth=2)
    plt.title('Утилизация CPU', fontsize=12)
    plt.ylabel('Загрузка CPU (0-1)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # График утилизации RAM
    plt.subplot(2, 1, 2)
    plt.plot(threshold_df['time'], threshold_df['ram_utilization'], 
             label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2)
    plt.plot(qlearning_df['time'], qlearning_df['ram_utilization'], 
             label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2)
    plt.plot(hybrid_df['time'], hybrid_df['ram_utilization'], 
             label='Гибридная модель', color=COLOR_HYBRID, linewidth=2)
    plt.title('Утилизация RAM', fontsize=12)
    plt.xlabel('Время (такты)', fontsize=10)
    plt.ylabel('Загрузка RAM (0-1)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Общий заголовок
    plt.suptitle(f'Утилизация ресурсов для сценария: {scenario_title}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'plots/{scenario_name}_resource_utilization.png', dpi=300, bbox_inches='tight')
    
    if DEBUG:
        print(f"График сохранен: plots/{scenario_name}_resource_utilization.png")
    
    plt.close()

def generate_summary_table():
    """
    Создание и вывод сводной таблицы со всеми средними значениями
    для всех моделей и всех сценариев
    """
    # Список всех summary файлов
    summary_files = find_summary_files()
    
    if not summary_files:
        print("Файлы с результатами не найдены")
        return None
    
    # Загружаем и объединяем все данные
    all_data = []
    for file in summary_files:
        try:
            data = pd.read_csv(file)
            all_data.append(data)
            if DEBUG:
                print(f"Загружены данные из {file}, {len(data)} строк")
        except Exception as e:
            print(f"Ошибка при чтении файла {file}: {e}")
    
    if not all_data:
        print("Не удалось загрузить данные из файлов")
        return None
    
    summary_df = pd.concat(all_data, ignore_index=True)
    
    # Фильтруем только нужные метрики
    key_metrics = ['avg_latency', 'avg_jitter', 'total_migrations', 'avg_cpu', 'avg_ram']
    
    # Если есть энергопотребление, добавляем его
    if any(summary_df['metric'] == 'avg_energy'):
        key_metrics.append('avg_energy')
    
    # Проверяем, есть ли нужные метрики
    metrics_found = set(summary_df['metric'].unique())
    if not any(metric in metrics_found for metric in key_metrics):
        print(f"Не найдены ключевые метрики в данных. Доступные метрики: {metrics_found}")
        return None
    
    filtered_df = summary_df[summary_df['metric'].isin(key_metrics)].copy()
    
    # Создаем сводную таблицу с обработкой дубликатов
    try:
        pivot_table = filtered_df.pivot_table(
            index=['scenario', 'model'],
            columns='metric',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        if DEBUG:
            print("Создана сводная таблица данных")
    except Exception as e:
        print(f"Ошибка при создании сводной таблицы: {e}")
        # Возможные проблемы с данными, попробуем показать их структуру
        if DEBUG:
            print("Структура доступных данных:")
            print(filtered_df.head())
            print("\nУникальные значения в ключевых столбцах:")
            print("Метрики:", filtered_df['metric'].unique())
            print("Сценарии:", filtered_df['scenario'].unique())
            print("Модели:", filtered_df['model'].unique())
        return None
    
    # Добавляем данные о разделении миграций для гибридной модели
    reactive_df = summary_df[summary_df['metric'] == 'reactive_migrations'].copy()
    proactive_df = summary_df[summary_df['metric'] == 'proactive_migrations'].copy()
    
    if not reactive_df.empty and not proactive_df.empty:
        # Добавляем колонки для реактивных и проактивных миграций
        reactive_pivot = reactive_df.pivot_table(
            index=['scenario', 'model'],
            values='value',
            aggfunc='mean'
        ).reset_index().rename(columns={'value': 'reactive_migrations'})
        
        proactive_pivot = proactive_df.pivot_table(
            index=['scenario', 'model'],
            values='value',
            aggfunc='mean'
        ).reset_index().rename(columns={'value': 'proactive_migrations'})
        
        # Объединяем с основной таблицей
        if not reactive_pivot.empty:
            pivot_table = pd.merge(
                pivot_table, reactive_pivot, 
                on=['scenario', 'model'], 
                how='left'
            )
        
        if not proactive_pivot.empty:
            pivot_table = pd.merge(
                pivot_table, proactive_pivot, 
                on=['scenario', 'model'], 
                how='left'
            )
    
    # Заменяем NaN на ""
    pivot_table = pivot_table.fillna("")
    
    # Форматируем названия моделей и сценариев
    model_display_names = {
        'threshold': 'Пороговая модель',
        'qlearning': 'Q-Learning модель',
        'hybrid': 'Гибридная модель'
    }
    
    scenario_display_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический',
        'limited_resources': 'Ограниченные ресурсы'
    }
    
    # Применяем форматирование
    pivot_table['model'] = pivot_table['model'].map(lambda x: model_display_names.get(x, x))
    pivot_table['scenario'] = pivot_table['scenario'].map(lambda x: scenario_display_names.get(x, x))
    
    # Переименовываем колонки для лучшей читаемости
    column_names = {
        'scenario': 'Сценарий',
        'model': 'Модель',
        'avg_latency': 'Средняя задержка (мс)',
        'avg_jitter': 'Средний джиттер (мс)',
        'total_migrations': 'Всего миграций',
        'avg_cpu': 'Средняя загрузка CPU',
        'avg_ram': 'Средняя загрузка RAM',
        'avg_energy': 'Среднее энергопотребление (Вт)',
        'reactive_migrations': 'Реактивные миграции',
        'proactive_migrations': 'Проактивные миграции'
    }
    
    # Применяем переименование колонок
    rename_columns = {}
    for col in pivot_table.columns:
        if col in column_names:
            rename_columns[col] = column_names[col]
    
    pivot_table = pivot_table.rename(columns=rename_columns)
    
    # Сохраняем таблицу в CSV
    os.makedirs('results', exist_ok=True)
    try:
        pivot_table.to_csv('results/comparative_results.csv', index=False)
        if DEBUG:
            print("Сводная таблица сохранена в results/comparative_results.csv")
    except Exception as e:
        print(f"Ошибка при сохранении сводной таблицы: {e}")
    
    # Возвращаем таблицу для вывода в консоль
    return pivot_table

def check_directories():
    """
    Проверка наличия нужных директорий и создание их при необходимости
    
    Возвращает:
    -----------
    bool : True, если все в порядке
    """
    # Создаем директории, если они не существуют
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Проверяем наличие файлов результатов
    results_files = glob.glob('results/*.csv')
    if not results_files:
        # Проверяем файлы в корневой директории
        root_files = glob.glob('*.csv')
        if root_files:
            if DEBUG:
                print(f"Файлы найдены в корневой директории: {len(root_files)}")
                print("Пытаемся переместить их в директорию results/...")
            
            # Перемещаем файлы в директорию results
            for file in root_files:
                try:
                    new_path = os.path.join('results', os.path.basename(file))
                    # Копируем файл (не перемещаем, чтобы не терять оригинал)
                    with open(file, 'r') as f_in:
                        with open(new_path, 'w') as f_out:
                            f_out.write(f_in.read())
                    if DEBUG:
                        print(f"Файл {file} скопирован в {new_path}")
                except Exception as e:
                    print(f"Ошибка при копировании файла {file}: {e}")
        else:
            print("Предупреждение: Файлы результатов не найдены ни в 'results/', ни в корневой директории.")
            print("Необходимо сначала запустить симуляцию.")
            return False
    
    return True

def main():
    """Основная функция для построения всех графиков"""
    # Проверяем наличие директорий и файлов
    if not check_directories():
        return
    
    # Список основных сценариев
    scenarios = ['critical', 'standard', 'mixed', 'dynamic', 'limited_resources']
    
    print("Генерация графиков и визуализаций...")
    
    # 1. График со всеми моделями и сценариями по количеству миграций
    plot_all_models_migrations()
    
    # 2. Графики производительности для каждого сценария
    for scenario in scenarios:
        # Графики задержки и джиттера
        plot_model_performance_per_scenario(scenario)
        
        # Графики энергопотребления
        plot_energy_comparison(scenario)
        
        # Графики утилизации ресурсов
        # plot_resource_utilization(scenario)
    
    # 3. Создание и вывод таблицы со всеми средними значениями
    print("\nСводная таблица результатов:")
    summary_table = generate_summary_table()
    
    # Вывод таблицы в консоль с красивым форматированием
    if summary_table is not None:
        try:
            print(tabulate(summary_table, headers='keys', tablefmt='grid', showindex=False))
        except Exception as e:
            print(f"Ошибка при форматировании таблицы: {e}")
            print(summary_table)
        
        print(f"\nТаблица также сохранена в файл: results/comparative_results.csv")
    
    print("Все графики сгенерированы в директории 'plots'.")


if __name__ == "__main__":
    main()
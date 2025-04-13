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
    с улучшенной обработкой непрерывности данных
    
    Параметры:
    ----------
    scenario_name : str
        Название сценария
    """
    if DEBUG:
        print(f"Обработка сценария: {scenario_name}")
    
    # Загружаем данные для всех моделей с обработкой ошибок
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
    
    # Проверяем наличие колонок
    expected_columns = ['time', 'latency', 'jitter']
    all_data_present = all(col in threshold_df.columns and col in qlearning_df.columns and col in hybrid_df.columns 
                          for col in expected_columns)
    
    if not all_data_present:
        print(f"Отсутствуют необходимые колонки для визуализации сценария {scenario_name}")
        return
    
    # Добавление скользящего среднего для сглаживания данных
    window_size = 5  # Размер окна для сглаживания
    
    # Применяем сглаживание для каждой модели
    for df in [threshold_df, qlearning_df, hybrid_df]:
        df['smooth_latency'] = df['latency'].rolling(window=window_size, min_periods=1).mean()
        df['smooth_jitter'] = df['jitter'].rolling(window=window_size, min_periods=1).mean()
        
        # Заполняем пропущенные значения линейной интерполяцией
        df['smooth_latency'] = df['smooth_latency'].interpolate(method='linear')
        df['smooth_jitter'] = df['smooth_jitter'].interpolate(method='linear')
        
        # Обработка краевых значений (начало и конец временного ряда)
        df['smooth_latency'] = df['smooth_latency'].fillna(method='bfill').fillna(method='ffill')
        df['smooth_jitter'] = df['smooth_jitter'].fillna(method='bfill').fillna(method='ffill')
    
    # Создаем фигуру с общей осью X для лучшей сопоставимости
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Общий диапазон данных для согласования осей Y
    all_latency = pd.concat([threshold_df['latency'], qlearning_df['latency'], hybrid_df['latency']])
    all_jitter = pd.concat([threshold_df['jitter'], qlearning_df['jitter'], hybrid_df['jitter']])
    
    # Исключаем выбросы для лучшего масштабирования (99 процентиль)
    latency_max = np.percentile(all_latency, 99)
    jitter_max = np.percentile(all_jitter, 99)
    
    # Определяем пороговые значения для обозначения на графиках
    latency_threshold = 50  # Пример порогового значения задержки (настройте при необходимости)
    jitter_threshold = 25   # Пример порогового значения джиттера (настройте при необходимости)
    
    # Цветовая схема для моделей
    latency_color = 'blue'
    jitter_color = 'red'
    reactive_migration_color = 'orange'
    proactive_migration_color = 'green'
    threshold_line_color = 'gray'
    
    # Функция для добавления отметок миграций оптимизированным способом
    def add_migration_markers(ax, df, model_type):
        # Группируем миграции по близости для уменьшения визуального шума
        if 'migration_performed' in df.columns and 'migration_success' in df.columns:
            successful_migrations = df[(df['migration_performed'] == True) & 
                                       (df['migration_success'] == True)]
            
            # Если это гибридная модель, разделяем типы миграций
            if model_type == 'hybrid' and 'migration_mode' in df.columns:
                reactive = successful_migrations[successful_migrations['migration_mode'] == 'reactive']
                proactive = successful_migrations[successful_migrations['migration_mode'] == 'proactive']
                
                # Добавляем полупрозрачные области вместо линий для групп миграций
                for t in reactive['time']:
                    ax.axvspan(t-0.5, t+0.5, alpha=0.2, color=reactive_migration_color)
                
                for t in proactive['time']:
                    ax.axvspan(t-0.5, t+0.5, alpha=0.2, color=proactive_migration_color)
                
                # Добавляем небольшие маркеры для индивидуальных миграций
                ax.scatter(reactive['time'], [latency_max * 0.1] * len(reactive), 
                          marker='|', color=reactive_migration_color, alpha=0.7, s=15,
                          label='Реактивные миграции')
                
                ax.scatter(proactive['time'], [latency_max * 0.1] * len(proactive), 
                          marker='|', color=proactive_migration_color, alpha=0.7, s=15,
                          label='Проактивные миграции')
            else:
                # Для не-гибридных моделей просто добавляем маркеры миграций
                for t in successful_migrations['time']:
                    ax.axvspan(t-0.5, t+0.5, alpha=0.2, color='green')
                
                ax.scatter(successful_migrations['time'], 
                          [latency_max * 0.1] * len(successful_migrations), 
                          marker='|', color='green', alpha=0.7, s=15,
                          label='Миграции')
    
    # Подграфик для пороговой модели
    ax1 = axes[0]
    ax1.plot(threshold_df['time'], threshold_df['smooth_latency'], 
            label='Задержка', color=latency_color, linewidth=2)
    ax1.plot(threshold_df['time'], threshold_df['smooth_jitter'], 
            label='Джиттер', color=jitter_color, linewidth=2)
    
    # Добавляем пороговые линии
    ax1.axhline(y=latency_threshold, color=threshold_line_color, linestyle='--', alpha=0.5)
    ax1.axhline(y=jitter_threshold, color=threshold_line_color, linestyle='-.', alpha=0.5)
    
    add_migration_markers(ax1, threshold_df, 'threshold')
    
    ax1.set_title(f'Задержка и джиттер: {model_display_names["threshold"]}', fontsize=12)
    ax1.set_ylabel('мс', fontsize=10)
    ax1.set_ylim(0, min(latency_max * 1.1, 200))  # Ограничиваем максимум для лучшей визуализации
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=9, loc='upper right')
    
    # Подграфик для Q-Learning модели
    ax2 = axes[1]
    ax2.plot(qlearning_df['time'], qlearning_df['smooth_latency'], 
            label='Задержка', color=latency_color, linewidth=2)
    ax2.plot(qlearning_df['time'], qlearning_df['smooth_jitter'], 
            label='Джиттер', color=jitter_color, linewidth=2)
    
    # Добавляем пороговые линии
    ax2.axhline(y=latency_threshold, color=threshold_line_color, linestyle='--', alpha=0.5)
    ax2.axhline(y=jitter_threshold, color=threshold_line_color, linestyle='-.', alpha=0.5)
    
    add_migration_markers(ax2, qlearning_df, 'qlearning')
    
    ax2.set_title(f'Задержка и джиттер: {model_display_names["qlearning"]}', fontsize=12)
    ax2.set_ylabel('мс', fontsize=10)
    ax2.set_ylim(0, min(latency_max * 1.1, 200))  # Ограничиваем максимум для лучшей визуализации
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=9, loc='upper right')
    
    # Подграфик для гибридной модели
    ax3 = axes[2]
    ax3.plot(hybrid_df['time'], hybrid_df['smooth_latency'], 
            label='Задержка', color=latency_color, linewidth=2)
    ax3.plot(hybrid_df['time'], hybrid_df['smooth_jitter'], 
            label='Джиттер', color=jitter_color, linewidth=2)
    
    # Добавляем пороговые линии
    ax3.axhline(y=latency_threshold, color=threshold_line_color, linestyle='--', alpha=0.5)
    ax3.axhline(y=jitter_threshold, color=threshold_line_color, linestyle='-.', alpha=0.5)
    
    add_migration_markers(ax3, hybrid_df, 'hybrid')
    
    ax3.set_title(f'Задержка и джиттер: {model_display_names["hybrid"]}', fontsize=12)
    ax3.set_xlabel('Время (такты)', fontsize=10)
    ax3.set_ylabel('мс', fontsize=10)
    ax3.set_ylim(0, min(latency_max * 1.1, 200))  # Ограничиваем максимум для лучшей визуализации
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=9, loc='upper right')
    
    # Добавляем аннотации с ключевыми статистиками
    def add_stats_annotations(ax, df, y_pos):
        avg_latency = df['latency'].mean()
        max_latency = df['latency'].max()
        avg_jitter = df['jitter'].mean()
        
        stats_text = f"Средняя задержка: {avg_latency:.2f} мс\n" \
                     f"Макс. задержка: {max_latency:.2f} мс\n" \
                     f"Средний джиттер: {avg_jitter:.2f} мс"
        
        # Размещаем текст в верхнем левом углу
        ax.text(0.02, y_pos, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'),
                fontsize=8, verticalalignment='top')
    
    # Добавляем аннотации для каждой модели
    add_stats_annotations(ax1, threshold_df, 0.98)
    add_stats_annotations(ax2, qlearning_df, 0.98)
    add_stats_annotations(ax3, hybrid_df, 0.98)
    
    # Общий заголовок
    plt.suptitle(f'Метрики задержки и джиттера для сценария: {scenario_title}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.25)
    
    # Добавляем общую легенду для миграций
    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color=latency_color, lw=2))
    labels.append('Задержка')
    handles.append(plt.Line2D([0], [0], color=jitter_color, lw=2))
    labels.append('Джиттер')
    handles.append(plt.Line2D([0], [0], color=threshold_line_color, linestyle='--', lw=1))
    labels.append('Пороговые значения')
    handles.append(plt.Line2D([0], [0], color='green', marker='|', lw=0, markersize=8))
    labels.append('Миграции')
    
    if any('migration_mode' in df.columns for df in [threshold_df, qlearning_df, hybrid_df]):
        handles.append(plt.Line2D([0], [0], color=reactive_migration_color, marker='|', lw=0, markersize=8))
        labels.append('Реактивные миграции')
        handles.append(plt.Line2D([0], [0], color=proactive_migration_color, marker='|', lw=0, markersize=8))
        labels.append('Проактивные миграции')
    
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01),
               fontsize=9, frameon=True, facecolor='white', edgecolor='gray')
    
    # Обновляем отступ снизу для размещения общей легенды
    plt.subplots_adjust(bottom=0.1)
    
    plt.savefig(f'plots/{scenario_name}_latency_jitter.png', dpi=300, bbox_inches='tight')
    
    if DEBUG:
        print(f"График сохранен: plots/{scenario_name}_latency_jitter.png")
    
    plt.close()
    
def plot_energy_comparison(scenario_name):
        """
        График сравнения энергопотребления между моделями с исправленными единицами
        
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
        
        # Проверяем наличие колонок
        if 'energy_consumption' not in threshold_df.columns or 'energy_consumption' not in qlearning_df.columns or 'energy_consumption' not in hybrid_df.columns:
            print(f"Отсутствуют данные об энергопотреблении для сценария {scenario_name}")
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
        plt.figure(figsize=(12, 8))
        
        # Создаем два подграфика
        # 1. Мгновенное энергопотребление
        plt.subplot(2, 1, 1)
        
        # Убедимся, что все модели видны, используем разные цвета и стили линий
        plt.plot(threshold_df['time'], threshold_df['energy_consumption'], 
                label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2, alpha=0.9)
        plt.plot(qlearning_df['time'], qlearning_df['energy_consumption'], 
                label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2, alpha=0.9)
        plt.plot(hybrid_df['time'], hybrid_df['energy_consumption'], 
                label='Гибридная модель', color=COLOR_HYBRID, linewidth=2, alpha=0.9)
        
        plt.title(f'Мгновенное энергопотребление', fontsize=12)
        plt.ylabel('Энергопотребление (кВт·ч)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 2. Кумулятивное энергопотребление
        plt.subplot(2, 1, 2)
        
        # Вычисляем кумулятивное энергопотребление
        threshold_cum = threshold_df['energy_consumption'].cumsum()
        qlearning_cum = qlearning_df['energy_consumption'].cumsum()
        hybrid_cum = hybrid_df['energy_consumption'].cumsum()
        
        plt.plot(threshold_df['time'], threshold_cum, 
                label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2, alpha=0.9)
        plt.plot(qlearning_df['time'], qlearning_cum, 
                label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2, alpha=0.9)
        plt.plot(hybrid_df['time'], hybrid_cum, 
                label='Гибридная модель', color=COLOR_HYBRID, linewidth=2, alpha=0.9)
        
        plt.title(f'Кумулятивное энергопотребление', fontsize=12)
        plt.xlabel('Время (такты)', fontsize=10)
        plt.ylabel('Суммарное энергопотребление (кВт·ч)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Общий заголовок
        plt.suptitle(f'Энергопотребление для сценария: {scenario_title}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'plots/{scenario_name}_energy.png', dpi=300, bbox_inches='tight')
        
        if DEBUG:
            print(f"График сохранен: plots/{scenario_name}_energy.png")
        
        plt.close()

# def plot_resource_utilization(scenario_name):
    # """
    # Создает графики утилизации CPU и RAM для всех моделей
    
    # Параметры:
    # ----------
    # scenario_name : str
    #     Название сценария
    # """
    # # Загружаем данные для всех моделей
    # threshold_df = load_results(scenario_name, 'threshold')
    # qlearning_df = load_results(scenario_name, 'qlearning')
    # hybrid_df = load_results(scenario_name, 'hybrid')
    
    # if threshold_df is None or qlearning_df is None or hybrid_df is None:
    #     print(f"Недостаточно данных для построения графиков утилизации ресурсов для сценария {scenario_name}")
    #     return
    
    # # Убедимся, что нужные колонки присутствуют
    # required_columns = ['time', 'cpu_utilization', 'ram_utilization']
    # if not all(col in threshold_df.columns for col in required_columns):
    #     print(f"Отсутствуют необходимые колонки в данных для сценария {scenario_name}")
    #     return
    
    # # Русские названия сценариев
    # scenario_display_names = {
    #     'critical': 'Критический',
    #     'standard': 'Стандартный',
    #     'mixed': 'Смешанный',
    #     'dynamic': 'Динамический',
    #     'limited_resources': 'Ограниченные ресурсы'
    # }
    
    # scenario_title = scenario_display_names.get(scenario_name, scenario_name)
    
    # # График утилизации ресурсов
    # plt.figure(figsize=(12, 8))
    
    # # График утилизации CPU
    # plt.subplot(2, 1, 1)
    # plt.plot(threshold_df['time'], threshold_df['cpu_utilization'], 
    #          label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2)
    # plt.plot(qlearning_df['time'], qlearning_df['cpu_utilization'], 
    #          label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2)
    # plt.plot(hybrid_df['time'], hybrid_df['cpu_utilization'], 
    #          label='Гибридная модель', color=COLOR_HYBRID, linewidth=2)
    # plt.title('Утилизация CPU', fontsize=12)
    # plt.ylabel('Загрузка CPU (0-1)', fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    
    # # График утилизации RAM
    # plt.subplot(2, 1, 2)
    # plt.plot(threshold_df['time'], threshold_df['ram_utilization'], 
    #          label='Пороговая модель', color=COLOR_THRESHOLD, linewidth=2)
    # plt.plot(qlearning_df['time'], qlearning_df['ram_utilization'], 
    #          label='Q-Learning модель', color=COLOR_QLEARNING, linewidth=2)
    # plt.plot(hybrid_df['time'], hybrid_df['ram_utilization'], 
    #          label='Гибридная модель', color=COLOR_HYBRID, linewidth=2)
    # plt.title('Утилизация RAM', fontsize=12)
    # plt.xlabel('Время (такты)', fontsize=10)
    # plt.ylabel('Загрузка RAM (0-1)', fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    
    # # Общий заголовок
    # plt.suptitle(f'Утилизация ресурсов для сценария: {scenario_title}', 
    #             fontsize=14, fontweight='bold')
    
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.92)
    # plt.savefig(f'plots/{scenario_name}_resource_utilization.png', dpi=300, bbox_inches='tight')
    
    # if DEBUG:
    #     print(f"График сохранен: plots/{scenario_name}_resource_utilization.png")
    
    # plt.close()

def plot_energy_breakdown(scenario_name):
    """
    График разбивки энергопотребления на базовое и миграционное для трёх моделей
    
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
    
    # Проверяем наличие необходимых колонок
    required_columns = ['baseline_energy', 'migration_energy']
    if not all(col in threshold_df.columns and col in qlearning_df.columns and col in hybrid_df.columns 
               for col in required_columns):
        print(f"Отсутствуют необходимые данные о разбивке энергопотребления для сценария {scenario_name}")
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
    
    # Получаем суммарные значения по категориям энергопотребления
    threshold_base = threshold_df['baseline_energy'].sum()
    threshold_migration = threshold_df['migration_energy'].sum()
    
    qlearning_base = qlearning_df['baseline_energy'].sum()
    qlearning_migration = qlearning_df['migration_energy'].sum()
    
    hybrid_base = hybrid_df['baseline_energy'].sum()
    hybrid_migration = hybrid_df['migration_energy'].sum()
    
    # Создаем график разбивки энергопотребления
    plt.figure(figsize=(12, 8))
    
    # Подготовка данных для графика
    models = ['Пороговая\nмодель', 'Q-Learning\nмодель', 'Гибридная\nмодель']
    base_energy = [threshold_base, qlearning_base, hybrid_base]
    migration_energy = [threshold_migration, qlearning_migration, hybrid_migration]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Создаем столбчатую диаграмму
    plt.bar(x, base_energy, width, label='Базовое энергопотребление', 
            color=['lightskyblue', 'lightgreen', 'lightcoral'])
    plt.bar(x, migration_energy, width, bottom=base_energy, 
            label='Энергопотребление миграций', 
            color=['steelblue', 'seagreen', 'indianred'])
    
    # Добавляем подписи процентов и абсолютных значений
    total_energy = [b + m for b, m in zip(base_energy, migration_energy)]
    
    for i in range(len(models)):
        base_pct = base_energy[i] / total_energy[i] * 100
        mig_pct = migration_energy[i] / total_energy[i] * 100
        
        # Подписи для базового энергопотребления
        plt.text(x[i], base_energy[i]/2, f"{base_pct:.1f}%", 
                 ha='center', va='center', color='black', fontweight='bold')
        
        # Подписи для миграционного энергопотребления
        plt.text(x[i], base_energy[i] + migration_energy[i]/2, f"{mig_pct:.1f}%", 
                 ha='center', va='center', color='black', fontweight='bold')
        
        # Общее энергопотребление
        plt.text(x[i], total_energy[i] + 0.5, f"{total_energy[i]:.2f} кВт·ч", 
                 ha='center', va='bottom')
    
    # Настраиваем отображение графика
    plt.title(f'Разбивка энергопотребления по компонентам для сценария: {scenario_title}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Энергопотребление (кВт·ч)', fontsize=12)
    plt.xticks(x, models, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(fontsize=10, loc='upper center')
    
    # Добавляем информацию об эффективности моделей
    plt.figtext(0.5, 0.01, 
               f"Общее энергопотребление: Пороговая модель = {total_energy[0]:.2f} кВт·ч, " +
               f"Q-Learning = {total_energy[1]:.2f} кВт·ч, Гибридная = {total_energy[2]:.2f} кВт·ч",
               ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'plots/{scenario_name}_energy_breakdown.png', dpi=300, bbox_inches='tight')
    
    if DEBUG:
        print(f"График сохранен: plots/{scenario_name}_energy_breakdown.png")
    
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
    
    # 2. Графики для каждого сценария
    for scenario in scenarios:
        # Графики задержки и джиттера
        plot_model_performance_per_scenario(scenario)
        
        # Графики энергопотребления
        plot_energy_comparison(scenario)
        
        # Новые графики разбивки энергопотребления
        plot_energy_breakdown(scenario)
    
    # 3. Создание и вывод сводной таблицы
    print("\nСводная таблица результатов:")
    summary_table = generate_summary_table()
    
    # Вывод таблицы в консоль
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
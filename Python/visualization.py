import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Установка русского шрифта для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

def visualize_scenario_results(scenario_type):
    """
    Визуализация результатов тестирования для заданного сценария
    
    Параметры:
    scenario_type - тип сценария ('critical', 'standard', 'mixed', 'dynamic')
    """
    # Перевод типов сценариев
    scenario_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический'
    }
    
    # Перевод названий моделей
    model_names = {
        'threshold': 'Пороговая',
        'qlearning': 'Q-learning',
        'hybrid': 'Гибридная'
    }
    
    # Загрузка данных из CSV-файлов
    latency_df = pd.read_csv(f"{scenario_type}_latency.csv")
    jitter_df = pd.read_csv(f"{scenario_type}_jitter.csv")
    energy_df = pd.read_csv(f"{scenario_type}_energy.csv")
    
    # Создание графиков
    plt.figure(figsize=(15, 10))
    
    # График задержки
    plt.subplot(3, 1, 1)
    for column in latency_df.columns:
        label = model_names.get(column, column)
        plt.plot(latency_df[column], label=label)
    
    # Добавление линии SLA для задержки
    sla_latency = 25  # Пример значения SLA для задержки (мс)
    plt.axhline(y=sla_latency, color='r', linestyle='--', label='SLA')
    
    plt.title(f'Задержка в сценарии "{scenario_names.get(scenario_type, scenario_type)}"')
    plt.xlabel('Время (такты)')
    plt.ylabel('Задержка (мс)')
    plt.legend()
    plt.grid(True)
    
    # График джиттера
    plt.subplot(3, 1, 2)
    for column in jitter_df.columns:
        label = model_names.get(column, column)
        plt.plot(jitter_df[column], label=label)
    
    # Добавление линии SLA для джиттера
    sla_jitter = 10  # Пример значения SLA для джиттера (мс)
    plt.axhline(y=sla_jitter, color='r', linestyle='--', label='SLA')
    
    plt.title(f'Джиттер в сценарии "{scenario_names.get(scenario_type, scenario_type)}"')
    plt.xlabel('Время (такты)')
    plt.ylabel('Джиттер (мс)')
    plt.legend()
    plt.grid(True)
    
    # График энергопотребления
    plt.subplot(3, 1, 3)
    for column in energy_df.columns:
        label = model_names.get(column, column)
        plt.plot(energy_df[column], label=label)
    
    plt.title(f'Энергопотребление в сценарии "{scenario_names.get(scenario_type, scenario_type)}"')
    plt.xlabel('Время (такты)')
    plt.ylabel('Энергопотребление (Вт)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{scenario_type}_results.png", dpi=300)
    plt.close()
    
    # Визуализация количества миграций
    try:
        migrations_df = pd.read_csv(f"{scenario_type}_migrations.csv")
        
        # График количества миграций
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        
        # Поиск всех моделей
        models = set()
        for column in migrations_df.columns:
            if "_migrations" in column or "_total" in column:
                models.add(column.split('_')[0])
        
        # Сбор данных о миграциях для каждой модели
        bar_positions = np.arange(len(models))
        
        # Построение графика для общего количества миграций
        migrations_data = []
        model_labels = []
        for model in models:
            model_labels.append(model_names.get(model, model))
            if f"{model}_migrations" in migrations_df.columns:
                migrations_data.append(migrations_df[f"{model}_migrations"].values[0])
            elif f"{model}_total" in migrations_df.columns:
                migrations_data.append(migrations_df[f"{model}_total"].values[0])
            else:
                migrations_data.append(0)
        
        plt.bar(bar_positions, migrations_data, bar_width, label='Всего миграций')
        
        # Для гибридной модели добавляем разбивку на реактивные и проактивные миграции
        reactive_data = []
        proactive_data = []
        has_detailed_data = False
        
        for model in models:
            if f"{model}_reactive" in migrations_df.columns and f"{model}_proactive" in migrations_df.columns:
                reactive_data.append(migrations_df[f"{model}_reactive"].values[0])
                proactive_data.append(migrations_df[f"{model}_proactive"].values[0])
                has_detailed_data = True
            else:
                reactive_data.append(0)
                proactive_data.append(0)
        
        if has_detailed_data:
            plt.bar(bar_positions + bar_width, reactive_data, bar_width, label='Реактивные миграции')
            plt.bar(bar_positions + 2 * bar_width, proactive_data, bar_width, label='Проактивные миграции')
        
        plt.xlabel('Модель миграции')
        plt.ylabel('Количество миграций')
        plt.title(f'Количество миграций в сценарии "{scenario_names.get(scenario_type, scenario_type)}"')
        plt.xticks(bar_positions + bar_width, model_labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{scenario_type}_migrations.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Не удалось визуализировать данные о миграциях для сценария {scenario_type}: {e}")

def visualize_all_scenarios():
    """Визуализация результатов для всех сценариев"""
    scenarios = ['critical', 'standard', 'mixed', 'dynamic']
    
    for scenario in scenarios:
        visualize_scenario_results(scenario)
    
    # Создание сводных графиков
    create_summary_graphs(scenarios)

def create_summary_graphs(scenarios):
    """Создание сводных графиков по всем сценариям"""
    # Перевод типов сценариев
    scenario_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический'
    }
    
    # Перевод названий моделей
    model_names = {
        'threshold': 'Пороговая',
        'qlearning': 'Q-обучение',
        'hybrid': 'Гибридная'
    }
    
    # Сбор средних значений метрик
    avg_metrics = {
        'latency': {'threshold': [], 'qlearning': [], 'hybrid': []},
        'jitter': {'threshold': [], 'qlearning': [], 'hybrid': []},
        'energy': {'threshold': [], 'qlearning': [], 'hybrid': []}
    }
    
    for scenario in scenarios:
        latency_df = pd.read_csv(f"{scenario}_latency.csv")
        jitter_df = pd.read_csv(f"{scenario}_jitter.csv")
        energy_df = pd.read_csv(f"{scenario}_energy.csv")
        
        for model in ['threshold', 'qlearning', 'hybrid']:
            if model in latency_df.columns:
                avg_metrics['latency'][model].append(latency_df[model].mean())
            if model in jitter_df.columns:
                avg_metrics['jitter'][model].append(jitter_df[model].mean())
            if model in energy_df.columns:
                avg_metrics['energy'][model].append(energy_df[model].mean())
    
    # Создание сводного графика средних значений
    plt.figure(figsize=(15, 10))
    
    # График средней задержки
    plt.subplot(3, 1, 1)
    bar_width = 0.25
    bar_positions = np.arange(len(scenarios))
    
    scenario_labels = [scenario_names.get(s, s) for s in scenarios]
    
    for i, model in enumerate(['threshold', 'qlearning', 'hybrid']):
        label = model_names.get(model, model)
        plt.bar(bar_positions + i * bar_width, avg_metrics['latency'][model], width=bar_width, label=label)
    
    plt.title('Средняя задержка по сценариям')
    plt.xlabel('Сценарий')
    plt.ylabel('Задержка (мс)')
    plt.xticks(bar_positions + bar_width, scenario_labels)
    plt.legend()
    plt.grid(True)
    
    # График среднего джиттера
    plt.subplot(3, 1, 2)
    
    for i, model in enumerate(['threshold', 'qlearning', 'hybrid']):
        label = model_names.get(model, model)
        plt.bar(bar_positions + i * bar_width, avg_metrics['jitter'][model], width=bar_width, label=label)
    
    plt.title('Средний джиттер по сценариям')
    plt.xlabel('Сценарий')
    plt.ylabel('Джиттер (мс)')
    plt.xticks(bar_positions + bar_width, scenario_labels)
    plt.legend()
    plt.grid(True)
    
    # График среднего энергопотребления
    plt.subplot(3, 1, 3)
    
    for i, model in enumerate(['threshold', 'qlearning', 'hybrid']):
        label = model_names.get(model, model)
        plt.bar(bar_positions + i * bar_width, avg_metrics['energy'][model], width=bar_width, label=label)
    
    plt.title('Среднее энергопотребление по сценариям')
    plt.xlabel('Сценарий')
    plt.ylabel('Энергопотребление (Вт)')
    plt.xticks(bar_positions + bar_width, scenario_labels)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("summary_results.png", dpi=300)
    plt.close()
    
    # Сбор данных о миграциях
    total_migrations = {
        'threshold': [],
        'qlearning': [],
        'hybrid': []
    }
    
    for scenario in scenarios:
        try:
            migrations_df = pd.read_csv(f"{scenario}_migrations.csv")
            
            for model in ['threshold', 'qlearning', 'hybrid']:
                if f"{model}_migrations" in migrations_df.columns:
                    total_migrations[model].append(migrations_df[f"{model}_migrations"].values[0])
                elif f"{model}_total" in migrations_df.columns:
                    total_migrations[model].append(migrations_df[f"{model}_total"].values[0])
                else:
                    total_migrations[model].append(0)
        except:
            for model in ['threshold', 'qlearning', 'hybrid']:
                total_migrations[model].append(0)
    
    # Создание сводного графика миграций
    plt.figure(figsize=(10, 6))
    
    for i, model in enumerate(['threshold', 'qlearning', 'hybrid']):
        label = model_names.get(model, model)
        plt.bar(bar_positions + i * bar_width, total_migrations[model], width=bar_width, label=label)
    
    plt.title('Общее количество миграций по сценариям')
    plt.xlabel('Сценарий')
    plt.ylabel('Количество миграций')
    plt.xticks(bar_positions + bar_width, scenario_labels)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("summary_migrations.png", dpi=300)
    plt.close()

def create_comparative_table():
    """Создание сравнительной таблицы моделей миграции"""
    scenarios = ['critical', 'standard', 'mixed', 'dynamic']
    models = ['threshold', 'qlearning', 'hybrid']
    metrics = ['latency', 'jitter', 'energy', 'migrations']
    
    # Перевод типов сценариев
    scenario_names = {
        'critical': 'Критический',
        'standard': 'Стандартный',
        'mixed': 'Смешанный',
        'dynamic': 'Динамический'
    }
    
    # Перевод названий моделей
    model_names = {
        'threshold': 'Пороговая',
        'qlearning': 'Q-обучение',
        'hybrid': 'Гибридная'
    }
    
    # Перевод названий метрик
    metric_names = {
        'latency': 'Задержка',
        'jitter': 'Джиттер',
        'energy': 'Энергопотребление',
        'migrations': 'Миграции'
    }
    
    results = {scenario: {model: {} for model in models} for scenario in scenarios}
    
    for scenario in scenarios:
        # Загрузка данных
        latency_df = pd.read_csv(f"{scenario}_latency.csv")
        jitter_df = pd.read_csv(f"{scenario}_jitter.csv")
        energy_df = pd.read_csv(f"{scenario}_energy.csv")
        
        try:
            migrations_df = pd.read_csv(f"{scenario}_migrations.csv")
        except:
            migrations_df = None
        
        for model in models:
            if model in latency_df.columns:
                results[scenario][model]['latency'] = latency_df[model].mean()
                results[scenario][model]['latency_std'] = latency_df[model].std()
            
            if model in jitter_df.columns:
                results[scenario][model]['jitter'] = jitter_df[model].mean()
                results[scenario][model]['jitter_std'] = jitter_df[model].std()
            
            if model in energy_df.columns:
                results[scenario][model]['energy'] = energy_df[model].mean()
                results[scenario][model]['energy_std'] = energy_df[model].std()
            
            if migrations_df is not None:
                if f"{model}_migrations" in migrations_df.columns:
                    results[scenario][model]['migrations'] = migrations_df[f"{model}_migrations"].values[0]
                elif f"{model}_total" in migrations_df.columns:
                    results[scenario][model]['migrations'] = migrations_df[f"{model}_total"].values[0]
    
    # Создание CSV с результатами
    rows = []
    
    for scenario in scenarios:
        for model in models:
            row = {'Сценарий': scenario_names.get(scenario, scenario), 
                   'Модель': model_names.get(model, model)}
            
            for metric in metrics:
                metric_name = metric_names.get(metric, metric)
                if metric in results[scenario][model]:
                    row[f'Среднее {metric_name}'] = results[scenario][model][metric]
                    
                    if f'{metric}_std' in results[scenario][model]:
                        row[f'СКО {metric_name}'] = results[scenario][model][f'{metric}_std']
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("comparative_results.csv", index=False)
    
    return df

# Пример использования:
# visualize_all_scenarios()
# comparative_table = create_comparative_table()
# print(comparative_table)
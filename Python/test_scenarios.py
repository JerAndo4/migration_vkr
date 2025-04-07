import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel
from hybrid_model import HybridModel

# Глобальные параметры симуляции
NUM_NODES = 4
NUM_SERVICES = 20
SIMULATION_TIME = 1000

class LoadScenario:
    """
    Базовый класс для сценариев нагрузки
    
    Параметры:
    ----------
    num_nodes : int
        Количество вычислительных узлов
    num_services : int
        Количество сервисов
    duration : int
        Длительность симуляции в тактах
    """
    
    def __init__(self, num_nodes=NUM_NODES, num_services=NUM_SERVICES, duration=SIMULATION_TIME):
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.duration = duration
        
        # Инициализация начальных метрик узлов и сервисов
        self.init_node_metrics = self._initialize_node_metrics()
        self.init_service_metrics = self._initialize_service_metrics()
        
    def _initialize_node_metrics(self):
        """
        Инициализация метрик узлов
        
        Возвращает:
        ----------
        dict
            Начальные метрики узлов
        """
        node_metrics = {}
        
        # Разная начальная загрузка для узлов
        for node_id in range(self.num_nodes):
            node_metrics[node_id] = {
                'cpu': 0.2 + 0.1 * node_id,  # Начальная загрузка CPU
                'ram': 0.2 + 0.05 * node_id,  # Начальная загрузка RAM
                'bandwidth': 10.0,  # Пропускная способность (Гбит/с)
                'latency': 5.0 + node_id * 2  # Задержка доступа (мс)
            }
            
        return node_metrics
    
    def _initialize_service_metrics(self):
        """
        Инициализация метрик сервисов
        
        Возвращает:
        ----------
        dict
            Начальные метрики сервисов
        """
        service_metrics = {}
        
        # Различные требования для сервисов
        for service_id in range(self.num_services):
            service_category = service_id % 5  # 5 категорий сервисов
            
            if service_category == 0:  # CPU-интенсивные сервисы
                cpu_usage = 0.15 + 0.05 * np.random.rand()
                ram_usage = 0.05 + 0.03 * np.random.rand()
                priority = 1  # Высший приоритет
            elif service_category == 1:  # RAM-интенсивные сервисы
                cpu_usage = 0.05 + 0.03 * np.random.rand()
                ram_usage = 0.15 + 0.05 * np.random.rand()
                priority = 2
            elif service_category == 2:  # Балансированные сервисы
                cpu_usage = 0.1 + 0.03 * np.random.rand()
                ram_usage = 0.1 + 0.03 * np.random.rand()
                priority = 3
            elif service_category == 3:  # Легкие сервисы
                cpu_usage = 0.03 + 0.02 * np.random.rand()
                ram_usage = 0.03 + 0.02 * np.random.rand()
                priority = 4
            else:  # Нерегулярные сервисы
                cpu_usage = 0.05 + 0.1 * np.random.rand()
                ram_usage = 0.05 + 0.1 * np.random.rand()
                priority = 5  # Низший приоритет
                
            service_metrics[service_id] = {
                'cpu': cpu_usage,
                'ram': ram_usage,
                'bandwidth': 0.5 + 1.5 * np.random.rand(),  # Требуемая пропускная способность (Гбит/с)
                'max_latency': 10 + 20 * np.random.rand(),  # Максимально допустимая задержка (мс)
                'priority': priority  # Приоритет сервиса (1 - наивысший)
            }
            
        return service_metrics
    
    def generate_workload(self):
        """
        Генерация рабочей нагрузки для симуляции
        
        Возвращает:
        ----------
        list
            Список метрик узлов и сервисов для каждого такта симуляции
        """
        # Базовая реализация - постоянная нагрузка
        workload = []
        
        for t in range(self.duration):
            node_metrics = self.init_node_metrics.copy()
            service_metrics = self.init_service_metrics.copy()
            
            workload.append({
                'time': t,
                'node_metrics': node_metrics,
                'service_metrics': service_metrics
            })
            
        return workload


class CriticalScenario(LoadScenario):
    """
    Критический сценарий с экстремальными нагрузками.
    Моделирует ситуацию с резкими скачками нагрузки на одном или 
    нескольких узлах, что требует быстрой реакции для 
    предотвращения отказов.
    """
    
    def generate_workload(self):
        workload = []
        
        for t in range(self.duration):
            # Копируем начальные метрики
            node_metrics = {node_id: metrics.copy() for node_id, metrics in self.init_node_metrics.items()}
            service_metrics = {service_id: metrics.copy() for service_id, metrics in self.init_service_metrics.items()}
            
            # Генерируем критические скачки нагрузки в определенные моменты времени
            if t >= 100 and t < 200:
                # Резкий скачок нагрузки на узле 0
                node_metrics[0]['cpu'] = min(1.0, node_metrics[0]['cpu'] + 0.5)
                node_metrics[0]['ram'] = min(1.0, node_metrics[0]['ram'] + 0.3)
            elif t >= 400 and t < 450:
                # Резкий скачок нагрузки на узле 1
                node_metrics[1]['cpu'] = min(1.0, node_metrics[1]['cpu'] + 0.6)
                node_metrics[1]['ram'] = min(1.0, node_metrics[1]['ram'] + 0.4)
            elif t >= 700 and t < 800:
                # Одновременная перегрузка узлов 2 и 3
                node_metrics[2]['cpu'] = min(1.0, node_metrics[2]['cpu'] + 0.4)
                node_metrics[2]['ram'] = min(1.0, node_metrics[2]['ram'] + 0.3)
                node_metrics[3]['cpu'] = min(1.0, node_metrics[3]['cpu'] + 0.5)
                node_metrics[3]['ram'] = min(1.0, node_metrics[3]['ram'] + 0.2)
            
            workload.append({
                'time': t,
                'node_metrics': node_metrics,
                'service_metrics': service_metrics
            })
            
        return workload


class StandardScenario(LoadScenario):
    """
    Стандартный сценарий с умеренной нагрузкой и небольшими колебаниями.
    Представляет типичные условия эксплуатации с плавными изменениями 
    и сезонными колебаниями.
    """
    
    def generate_workload(self):
        workload = []
        
        for t in range(self.duration):
            # Копируем начальные метрики
            node_metrics = {node_id: metrics.copy() for node_id, metrics in self.init_node_metrics.items()}
            service_metrics = {service_id: metrics.copy() for service_id, metrics in self.init_service_metrics.items()}
            
            # Дневные колебания нагрузки (период 100 тактов)
            daily_factor = 0.2 * np.sin(2 * np.pi * t / 100)
            
            # Недельные колебания (период 700 тактов)
            weekly_factor = 0.1 * np.sin(2 * np.pi * t / 700)
            
            # Случайные колебания
            random_factor = 0.05 * np.random.randn()
            
            # Применяем факторы к метрикам узлов
            for node_id in range(self.num_nodes):
                node_variation = 0.1 * np.sin(2 * np.pi * (t + 25 * node_id) / 100)  # Разные фазы для разных узлов
                load_change = daily_factor + weekly_factor + random_factor + node_variation
                
                node_metrics[node_id]['cpu'] = max(0.1, min(0.9, node_metrics[node_id]['cpu'] + load_change))
                node_metrics[node_id]['ram'] = max(0.1, min(0.9, node_metrics[node_id]['ram'] + 0.7 * load_change))
            
            workload.append({
                'time': t,
                'node_metrics': node_metrics,
                'service_metrics': service_metrics
            })
            
        return workload


class MixedScenario(LoadScenario):
    """
    Смешанный сценарий с чередующимися периодами стабильной и высокой нагрузки.
    Моделирует смену паттернов использования сети с перемещением нагрузки 
    между разными зонами обслуживания. Проверяет адаптивность моделей к 
    изменяющимся условиям.
    """
    
    def generate_workload(self):
        workload = []
        
        # Определяем периоды активной нагрузки для каждого узла
        active_periods = {
            0: [(100, 200), (600, 700)],
            1: [(200, 300), (700, 800)],
            2: [(300, 400), (800, 900)],
            3: [(400, 500), (900, 1000)]
        }
        
        for t in range(self.duration):
            # Копируем начальные метрики
            node_metrics = {node_id: metrics.copy() for node_id, metrics in self.init_node_metrics.items()}
            service_metrics = {service_id: metrics.copy() for service_id, metrics in self.init_service_metrics.items()}
            
            # Применяем нагрузку для каждого узла
            for node_id in range(self.num_nodes):
                # Проверяем, находится ли текущее время в активном периоде для узла
                is_active = any(start <= t < end for start, end in active_periods[node_id])
                
                if is_active:
                    # Высокая нагрузка для активного периода (гарантированно превышает порог)
                    load_increase = 0.5 + 0.2 * np.sin(2 * np.pi * t / 50)
                    node_metrics[node_id]['cpu'] = min(0.95, node_metrics[node_id]['cpu'] + load_increase)
                    node_metrics[node_id]['ram'] = min(0.95, node_metrics[node_id]['ram'] + 0.8 * load_increase)
                else:
                    # Нормальная нагрузка для неактивного периода
                    load_variation = 0.1 * np.sin(2 * np.pi * t / 100) + 0.05 * np.random.randn()
                    node_metrics[node_id]['cpu'] = max(0.1, min(0.5, node_metrics[node_id]['cpu'] + load_variation))
                    node_metrics[node_id]['ram'] = max(0.1, min(0.5, node_metrics[node_id]['ram'] + load_variation))
            
            workload.append({
                'time': t,
                'node_metrics': node_metrics,
                'service_metrics': service_metrics
            })
            
        return workload


class DynamicScenario(LoadScenario):
    """
    Динамический сценарий с комплексными паттернами нагрузки.
    Имитирует реальные условия эксплуатации с трендами, сезонностью и 
    случайными компонентами. Наиболее сложный сценарий для проверки моделей.
    """
    
    def generate_workload(self):
        workload = []
        
        # Комбинируем несколько паттернов нагрузки
        for t in range(self.duration):
            # Копируем начальные метрики
            node_metrics = {node_id: metrics.copy() for node_id, metrics in self.init_node_metrics.items()}
            service_metrics = {service_id: metrics.copy() for service_id, metrics in self.init_service_metrics.items()}
            
            # Базовый тренд (постепенный рост нагрузки)
            trend_factor = 0.3 * t / self.duration
            
            # Периодические колебания разной частоты
            daily_factor = 0.2 * np.sin(2 * np.pi * t / 100)
            weekly_factor = 0.15 * np.sin(2 * np.pi * t / 700)
            
            # Случайные выбросы (с малой вероятностью)
            spike_factor = 0 if np.random.rand() > 0.02 else 0.3 * np.random.rand()
            
            # Применяем факторы к узлам с разными весами
            for node_id in range(self.num_nodes):
                node_weight = 1.0 + 0.2 * node_id  # Разные веса для разных узлов
                
                # Узлоспецифичные условия
                if node_id == 0 and 200 <= t < 300:
                    # Сценарий миграции для узла 0
                    node_specific = 0.4 * np.sin(np.pi * (t - 200) / 100)
                elif node_id == 1 and 500 <= t < 600:
                    # Пиковая нагрузка для узла 1
                    node_specific = 0.3
                elif node_id == 2 and 300 <= t < 400:
                    # Снижение нагрузки для узла 2
                    node_specific = -0.2
                elif node_id == 3 and 700 <= t < 800:
                    # Переменная нагрузка для узла 3
                    node_specific = 0.2 * np.sin(4 * np.pi * (t - 700) / 100)
                else:
                    node_specific = 0
                
                # Комбинированное изменение нагрузки
                load_change = (trend_factor + daily_factor + weekly_factor + spike_factor + node_specific) * node_weight
                
                # Применяем изменение к метрикам узла
                node_metrics[node_id]['cpu'] = max(0.1, min(0.95, node_metrics[node_id]['cpu'] + load_change))
                node_metrics[node_id]['ram'] = max(0.1, min(0.95, node_metrics[node_id]['ram'] + 0.8 * load_change))
            
            workload.append({
                'time': t,
                'node_metrics': node_metrics,
                'service_metrics': service_metrics
            })
            
        return workload


class LimitedResourcesScenario(LoadScenario):
    """
    Сценарий с ограниченными ресурсами для миграции.
    Тестирует работу моделей в условиях, когда все узлы имеют высокую
    базовую нагрузку и существует дефицит свободных ресурсов для миграции.
    """
    
    def generate_workload(self):
        workload = []
        
        for t in range(self.duration):
            # Копируем начальные метрики
            node_metrics = {node_id: metrics.copy() for node_id, metrics in self.init_node_metrics.items()}
            service_metrics = {service_id: metrics.copy() for service_id, metrics in self.init_service_metrics.items()}
            
            # Высокая базовая нагрузка на всех узлах
            for node_id in range(self.num_nodes):
                node_metrics[node_id]['cpu'] = 0.6 + 0.05 * node_id
                node_metrics[node_id]['ram'] = 0.65 + 0.03 * node_id
            
            # Периодические колебания и случайные факторы
            time_factor = 0.15 * np.sin(2 * np.pi * t / 200)
            random_factor = 0.05 * np.random.randn()
            
            # Пиковые нагрузки на разных узлах в разное время
            for node_id in range(self.num_nodes):
                # Создаем перегрузку на каждом узле в определенный период
                peak_period = 150 + 200 * node_id
                if peak_period <= t < peak_period + 100:
                    peak_factor = 0.25 * np.sin(np.pi * (t - peak_period) / 100)
                else:
                    peak_factor = 0
                
                load_change = time_factor + random_factor + peak_factor
                
                node_metrics[node_id]['cpu'] = max(0.5, min(0.95, node_metrics[node_id]['cpu'] + load_change))
                node_metrics[node_id]['ram'] = max(0.5, min(0.95, node_metrics[node_id]['ram'] + 0.7 * load_change))
            
            workload.append({
                'time': t,
                'node_metrics': node_metrics,
                'service_metrics': service_metrics
            })
            
        return workload


def run_simulation(model, workload, scenario_name):
    """
    Запуск симуляции модели на заданной рабочей нагрузке
    
    Параметры:
    ----------
    model : ThresholdModel, QLearningModel или HybridModel
        Модель миграции
    workload : list
        Рабочая нагрузка, сгенерированная сценарием
    scenario_name : str
        Название сценария
        
    Возвращает:
    ----------
    dict
        Результаты симуляции
    """
    results = {
        'time': [],
        'overloaded_nodes': [],
        'predicted_overloads': [],
        'migrations': [],
        'cpu_utilization': [],
        'ram_utilization': [],
        'latency': [],
        'jitter': [],
        'energy_consumption': []  # Добавляем энергопотребление
    }
    
    # Для гибридной модели добавляем отслеживание типов миграций
    if isinstance(model, HybridModel):
        results['reactive_migrations'] = []
        results['proactive_migrations'] = []
    
    # Средние значения задержки для расчета джиттера
    latency_window = []
    
    # Общее количество миграций для отладки
    total_migrations = 0
    
    for step_data in workload:
        time = step_data['time']
        node_metrics = step_data['node_metrics']
        service_metrics = step_data['service_metrics']
        
        # Выполняем шаг моделирования
        step_results = model.step(node_metrics, service_metrics)
        
        # Сохраняем базовые метрики
        results['time'].append(time)
        
        # Количество миграций на данном шаге
        migrations_count = len(step_results['migrations'])
        results['migrations'].append(migrations_count)
        total_migrations += migrations_count
        
        results['overloaded_nodes'].append(len(step_results.get('overloaded_nodes', [])))
        results['predicted_overloads'].append(len(step_results.get('predicted_overloads', [])))
        
        # Для гибридной модели отслеживаем типы миграций
        if isinstance(model, HybridModel):
            # Подсчитываем реактивные и проактивные миграции
            reactive_count = len([m for m in step_results['migrations'] if m.get('type') == 'reactive'])
            proactive_count = len([m for m in step_results['migrations'] if m.get('type') == 'proactive'])
            
            # Проверяем сумму типов миграций
            if reactive_count + proactive_count != migrations_count:
                print(f"Предупреждение: Несоответствие в типах миграций для гибридной модели шаг {time}.")
                print(f"  Всего: {migrations_count}, Реактивные: {reactive_count}, Проактивные: {proactive_count}")
            
            results['reactive_migrations'].append(reactive_count)
            results['proactive_migrations'].append(proactive_count)
        
        # Рассчитываем метрики производительности
        avg_cpu = np.mean([metrics['cpu'] for metrics in node_metrics.values()])
        avg_ram = np.mean([metrics['ram'] for metrics in node_metrics.values()])
        results['cpu_utilization'].append(avg_cpu)
        results['ram_utilization'].append(avg_ram)
        
        # Расчет средней задержки на основе размещения сервисов
        total_latency = 0
        service_count = 0
        
        for service_id, node_id in model.service_placement.items():
            # Задержка зависит от узла и загрузки узла
            base_latency = node_metrics[node_id]['latency']
            load_factor = 1 + node_metrics[node_id]['cpu']  # Нагрузка увеличивает задержку
            service_latency = base_latency * load_factor
            total_latency += service_latency
            service_count += 1
        
        avg_latency = total_latency / service_count if service_count > 0 else 0
        results['latency'].append(avg_latency)
        
        # Расчет джиттера (вариации задержки)
        latency_window.append(avg_latency)
        if len(latency_window) > 10:  # Окно для расчета джиттера
            latency_window.pop(0)
        
        jitter = np.std(latency_window) if len(latency_window) > 1 else 0
        results['jitter'].append(jitter)
        
        # Расчет энергопотребления (зависит от загрузки CPU и количества узлов)
        # Модель: базовое потребление + дополнительное в зависимости от загрузки CPU
        base_energy_per_node = 100  # Базовое потребление в ваттах на узел
        max_additional_energy = 150  # Максимальное дополнительное потребление при полной загрузке
        
        total_energy = 0
        for node_id, metrics in node_metrics.items():
            node_energy = base_energy_per_node + (metrics['cpu'] * max_additional_energy)
            total_energy += node_energy
        
        results['energy_consumption'].append(total_energy)
    
    # Проверка и вывод итоговой информации о миграциях
    print(f"      Общее количество миграций (по шагам): {total_migrations}")
    
    if isinstance(model, HybridModel):
        total_reactive = sum(results['reactive_migrations'])
        total_proactive = sum(results['proactive_migrations'])
        print(f"      Реактивные миграции: {total_reactive}")
        print(f"      Проактивные миграции: {total_proactive}")
        print(f"      Всего миграций (реактивные + проактивные): {total_reactive + total_proactive}")
    
    # Сохраняем результаты в CSV
    save_results_to_csv(results, scenario_name, model.__class__.__name__)
    
    return results


def save_results_to_csv(results, scenario_name, model_name):
    """
    Сохранение результатов симуляции в CSV-файл
    
    Параметры:
    ----------
    results : dict
        Результаты симуляции
    scenario_name : str
        Название сценария
    model_name : str
        Название модели
    """
    # Создаем директорию для результатов, если она не существует
    os.makedirs('results', exist_ok=True)
    
    # Приводим названия моделей к стандартному виду
    if model_name == 'ThresholdModel':
        model_short_name = 'threshold'
    elif model_name == 'QLearningModel':
        model_short_name = 'qlearning'
    elif model_name == 'HybridModel':
        model_short_name = 'hybrid'
    else:
        model_short_name = model_name.lower()
    
    # Формируем имя файла
    filename = f'results/{scenario_name}_{model_short_name}.csv'
    
    # Записываем результаты в CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(results['time'])):
            row = {field: results[field][i] for field in fieldnames}
            writer.writerow(row)
    
    # Создаем сводный файл для сравнения моделей
    summary_filename = f'results/{scenario_name}_summary.csv'
    
    # Проверяем, существует ли файл
    if not os.path.exists(summary_filename):
        with open(summary_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['scenario', 'model', 'metric', 'value'])
    
    # Добавляем сводные метрики
    with open(summary_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Средние значения ключевых метрик
        metrics = {
            'avg_migrations': np.mean(results['migrations']),
            'total_migrations': np.sum(results['migrations']),
            'avg_latency': np.mean(results['latency']),
            'avg_jitter': np.mean(results['jitter']),
            'avg_cpu': np.mean(results['cpu_utilization']),
            'avg_ram': np.mean(results['ram_utilization']),
            'avg_energy': np.mean(results['energy_consumption']),
            'max_latency': np.max(results['latency']),
            'max_jitter': np.max(results['jitter'])
        }
        
        # Для гибридной модели добавляем метрики разделения миграций на реактивные и проактивные
        if model_short_name == 'hybrid':
            # Проверяем, есть ли информация о типах миграций
            reactive_migrations = 0
            proactive_migrations = 0
            
            # Пытаемся получить информацию из метрик истории гибридной модели
            if 'reactive_migrations' in results and 'proactive_migrations' in results:
                reactive_migrations = np.sum(results['reactive_migrations'])
                proactive_migrations = np.sum(results['proactive_migrations'])
            
            # Если нет данных о разделении, оцениваем как 50/50
            if reactive_migrations == 0 and proactive_migrations == 0:
                total_migrations = metrics['total_migrations']
                reactive_migrations = total_migrations * 0.5
                proactive_migrations = total_migrations * 0.5
            
            metrics['reactive_migrations'] = reactive_migrations
            metrics['proactive_migrations'] = proactive_migrations
        
        for metric, value in metrics.items():
            writer.writerow([scenario_name, model_short_name, metric, value])


def run_scenarios():
    """Запуск всех сценариев для всех моделей"""
    # Сценарии с обновленными названиями
    scenarios = {
        'critical': CriticalScenario(),
        'standard': StandardScenario(),
        'mixed': MixedScenario(),  # Переименованный сценарий (был UserMigrationScenario)
        'dynamic': DynamicScenario(),
        'limited_resources': LimitedResourcesScenario()
    }
    
    # Модели
    models = {
        'threshold': ThresholdModel(),
        'qlearning': QLearningModel(),
        'hybrid': HybridModel()
    }
    
    # Запускаем все сценарии для всех моделей
    for scenario_name, scenario in scenarios.items():
        print(f"Запуск сценария: {scenario_name}")
        
        # Генерируем нагрузку для сценария
        workload = scenario.generate_workload()
        
        for model_name, model in models.items():
            print(f"  Тестирование модели: {model_name}")
            
            # Создаем новый экземпляр модели для каждого сценария
            if model_name == 'threshold':
                test_model = ThresholdModel()
            elif model_name == 'qlearning':
                test_model = QLearningModel()
            else:
                test_model = HybridModel()
            
            # Запускаем симуляцию
            results = run_simulation(test_model, workload, scenario_name)
            
            print(f"    Завершено с {sum(results['migrations'])} миграциями")
    
    print("Все сценарии завершены. Результаты сохранены в директории 'results'.")


# Запуск сценариев при выполнении скрипта напрямую
if __name__ == "__main__":
    run_scenarios()
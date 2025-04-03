import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Константы для симуляции
NUM_NODES = 10
NUM_SERVICES = 20
SIMULATION_TIME = 1000

class NetworkSimulator:
    def __init__(self, scenario_type):
        """
        Инициализация симулятора сети
        
        Параметры:
        scenario_type - тип сценария нагрузки ('critical', 'standard', 'mixed', 'dynamic')
        """
        self.scenario_type = scenario_type
        self.simulation_time = SIMULATION_TIME
        self.current_time = 0
        
        # Инициализация узлов
        self.nodes = self._initialize_nodes()
        
        # Инициализация сервисов
        self.services = self._initialize_services()
        
        # Начальное размещение сервисов
        self.service_placement = self._initial_service_placement()
        
        # Хранение метрик
        self.metrics = {
            'latency': np.zeros(SIMULATION_TIME),
            'jitter': np.zeros(SIMULATION_TIME),
            'energy': np.zeros(SIMULATION_TIME),
            'node_cpu_usage': np.zeros((SIMULATION_TIME, NUM_NODES)),
            'node_memory_usage': np.zeros((SIMULATION_TIME, NUM_NODES))
        }
    
    def _initialize_nodes(self):
        """Инициализация вычислительных узлов"""
        nodes = []
        
        # Облачный узел с высокой мощностью
        cloud_node = {
            'id': 0,
            'type': 'cloud',
            'cpu_capacity': 100,  # Условных единиц CPU
            'memory_capacity': 256,  # ГБ RAM
            'energy_consumption_idle': 100,  # Вт в режиме простоя
            'energy_consumption_max': 500,  # Вт при полной нагрузке
            'location': (0, 0)  # Условные координаты
        }
        nodes.append(cloud_node)
        
        # Граничные узлы со средней мощностью
        for i in range(1, 4):
            edge_node = {
                'id': i,
                'type': 'edge',
                'cpu_capacity': 50,
                'memory_capacity': 128,
                'energy_consumption_idle': 50,
                'energy_consumption_max': 200,
                'location': (i * 10, i * 10)
            }
            nodes.append(edge_node)
        
        # Узлы доступа с низкой мощностью
        for i in range(4, NUM_NODES):
            access_node = {
                'id': i,
                'type': 'access',
                'cpu_capacity': 25,
                'memory_capacity': 64,
                'energy_consumption_idle': 25,
                'energy_consumption_max': 100,
                'location': (i * 5, i * 5)
            }
            nodes.append(access_node)
        
        return nodes
    
    def _initialize_services(self):
        """Инициализация сервисов"""
        services = []
        
        # Сервисы с высокими требованиями (критические)
        for i in range(5):
            service = {
                'id': i,
                'type': 'critical',
                'cpu_requirement_base': 10,  # Базовые требования CPU
                'memory_requirement_base': 16,  # Базовые требования памяти в ГБ
                'latency_requirement': 10  # Максимально допустимая задержка в мс
            }
            services.append(service)
        
        # Сервисы со средними требованиями
        for i in range(5, 15):
            service = {
                'id': i,
                'type': 'standard',
                'cpu_requirement_base': 5,
                'memory_requirement_base': 8,
                'latency_requirement': 50
            }
            services.append(service)
        
        # Сервисы с низкими требованиями
        for i in range(15, NUM_SERVICES):
            service = {
                'id': i,
                'type': 'background',
                'cpu_requirement_base': 2,
                'memory_requirement_base': 4,
                'latency_requirement': 100
            }
            services.append(service)
        
        return services
    
    def _initial_service_placement(self):
        """Начальное размещение сервисов на узлах"""
        service_placement = {}
        
        # Критические сервисы размещаем на облачном узле
        for i in range(5):
            service_placement[i] = 0
        
        # Стандартные сервисы распределяем между граничными узлами
        for i in range(5, 15):
            service_placement[i] = 1 + (i - 5) % 3
        
        # Фоновые сервисы распределяем между узлами доступа
        for i in range(15, NUM_SERVICES):
            service_placement[i] = 4 + (i - 15) % 6
        
        return service_placement
    
    def _calculate_node_metrics(self):
        """Расчет текущих метрик узлов (загрузка CPU и памяти)"""
        node_metrics = {}
        
        # Инициализация метрик для каждого узла
        for node in self.nodes:
            node_id = node['id']
            node_metrics[node_id] = {
                'cpu_usage': 0.0,  # Доля использования от 0 до 1
                'memory_usage': 0.0,  # Доля использования от 0 до 1
                'energy_consumption': node['energy_consumption_idle']  # Начинаем с энергопотребления в режиме простоя
            }
        
        # Расчет загрузки от сервисов
        for service_id, node_id in self.service_placement.items():
            service = self.services[service_id]
            node = self.nodes[node_id]
            
            # Расчет текущих требований сервиса (с учетом нагрузки)
            cpu_requirement = self._get_service_cpu_requirement(service_id)
            memory_requirement = self._get_service_memory_requirement(service_id)
            
            # Обновление метрик узла
            node_metrics[node_id]['cpu_usage'] += cpu_requirement / node['cpu_capacity']
            node_metrics[node_id]['memory_usage'] += memory_requirement / node['memory_capacity']
            
            # Расчет дополнительного энергопотребления
            cpu_usage_ratio = node_metrics[node_id]['cpu_usage']
            energy_range = node['energy_consumption_max'] - node['energy_consumption_idle']
            node_metrics[node_id]['energy_consumption'] = node['energy_consumption_idle'] + (energy_range * cpu_usage_ratio)
        
        return node_metrics
    
    def _get_service_cpu_requirement(self, service_id):
        """Расчет текущих требований сервиса к CPU с учетом нагрузки"""
        service = self.services[service_id]
        base_requirement = service['cpu_requirement_base']
        
        # Применение модификатора нагрузки в зависимости от сценария
        load_factor = self._get_load_factor(service_id)
        
        return base_requirement * load_factor
    
    def _get_service_memory_requirement(self, service_id):
        """Расчет текущих требований сервиса к памяти с учетом нагрузки"""
        service = self.services[service_id]
        base_requirement = service['memory_requirement_base']
        
        # Применение модификатора нагрузки в зависимости от сценария
        load_factor = self._get_load_factor(service_id)
        
        # Память менее волатильна, чем CPU
        memory_factor = 1.0 + (load_factor - 1.0) * 0.5
        
        return base_requirement * memory_factor
    
    def _get_load_factor(self, service_id):
        """Расчет фактора нагрузки для сервиса в зависимости от сценария и времени"""
        service = self.services[service_id]
        time = self.current_time
        
        if self.scenario_type == 'critical':
            # Критический сценарий: высокая нагрузка на критические сервисы
            if service['type'] == 'critical':
                return 1.5 + 0.5 * np.sin(time / 50.0)  # Колебания от 1.0 до 2.0
            else:
                return 1.0 + 0.2 * np.sin(time / 100.0)  # Небольшие колебания
        
        elif self.scenario_type == 'standard':
            # Стандартный сценарий: умеренная нагрузка на все сервисы
            return 0.8 + 0.2 * np.sin(time / 100.0)  # Колебания от 0.6 до 1.0
        
        elif self.scenario_type == 'mixed':
            # Смешанный сценарий: чередование стабильной и высокой нагрузки
            if (time // 200) % 2 == 0:  # Периоды по 200 временных тактов
                return 0.7 + 0.2 * np.sin(time / 50.0)  # Стабильный период
            else:
                return 1.2 + 0.4 * np.sin(time / 30.0)  # Период высокой нагрузки
        
        elif self.scenario_type == 'dynamic':
            # Динамический сценарий: сложные паттерны нагрузки
            # Долгосрочный тренд
            trend = 0.3 * np.sin(time / 300.0)
            # Сезонный компонент
            season = 0.2 * np.sin(time / 50.0)
            # Случайная составляющая
            noise = 0.1 * np.random.randn()
            # Базовая нагрузка в зависимости от типа сервиса
            if service['type'] == 'critical':
                base = 1.2
            elif service['type'] == 'standard':
                base = 1.0
            else:
                base = 0.8
            
            return max(0.1, min(2.0, base + trend + season + noise))
        
        else:
            # По умолчанию стабильная нагрузка
            return 1.0
    
    def _calculate_latency(self):
        """Расчет средней задержки в сети"""
        total_weighted_latency = 0
        total_weight = 0
        
        node_metrics = self._calculate_node_metrics()
        
        for service_id, node_id in self.service_placement.items():
            service = self.services[service_id]
            
            # Базовая задержка в зависимости от типа узла
            node_type = self.nodes[node_id]['type']
            if node_type == 'cloud':
                base_latency = 30  # мс
            elif node_type == 'edge':
                base_latency = 15  # мс
            else:  # 'access'
                base_latency = 5  # мс
            
            # Дополнительная задержка при высокой загрузке
            additional_latency = max(0, (node_metrics[node_id]['cpu_usage'] - 0.8) * 100) if node_metrics[node_id]['cpu_usage'] > 0.8 else 0
            
            # Общая задержка для сервиса
            service_latency = base_latency + additional_latency
            
            # Вес в зависимости от типа сервиса
            if service['type'] == 'critical':
                weight = 3
            elif service['type'] == 'standard':
                weight = 2
            else:  # 'background'
                weight = 1
            
            total_weighted_latency += service_latency * weight
            total_weight += weight
        
        return total_weighted_latency / total_weight if total_weight > 0 else 0
    
    def _calculate_jitter(self):
        """Расчет джиттера (вариации задержки)"""
        if self.current_time < 2:
            return 0
        
        current_latency = self.metrics['latency'][self.current_time - 1]
        previous_latency = self.metrics['latency'][self.current_time - 2]
        
        return abs(current_latency - previous_latency)
    
    def _calculate_energy(self, node_metrics):
        """Расчет общего энергопотребления"""
        total_energy = sum(metrics['energy_consumption'] for metrics in node_metrics.values())
        return total_energy
    
    def _calculate_load_balance(self, node_metrics):
        """Расчет показателя балансировки нагрузки"""
        cpu_usages = [metrics['cpu_usage'] for metrics in node_metrics.values()]
        return 1.0 - np.std(cpu_usages)  # Более высокое значение соответствует лучшей балансировке
    
    def _update_metrics(self, node_metrics):
        """Обновление метрик производительности"""
        # Расчет задержки
        latency = self._calculate_latency()
        self.metrics['latency'][self.current_time] = latency
        
        # Расчет джиттера
        jitter = self._calculate_jitter()
        self.metrics['jitter'][self.current_time] = jitter
        
        # Расчет энергопотребления
        energy = self._calculate_energy(node_metrics)
        self.metrics['energy'][self.current_time] = energy
        
        # Сохранение загрузки узлов
        for node_id, metrics in node_metrics.items():
            self.metrics['node_cpu_usage'][self.current_time, node_id] = metrics['cpu_usage']
            self.metrics['node_memory_usage'][self.current_time, node_id] = metrics['memory_usage']
    
    def run_simulation(self, migration_model):
        """Запуск симуляции с заданной моделью миграции"""
        self.current_time = 0
        
        # Основной цикл симуляции
        for t in range(SIMULATION_TIME):
            self.current_time = t
            
            # Расчет текущих метрик узлов
            node_metrics = self._calculate_node_metrics()
            
            # Вызов модели миграции для принятия решений
            migrations = migration_model.migrate(node_metrics, self.service_placement)
            
            # Выполнение миграций
            for service_id, source_node, target_node in migrations:
                self.service_placement[service_id] = target_node
            
            # Обновление метрик производительности
            self._update_metrics(node_metrics)
            
            # Обновление метрик модели миграции
            migration_model.update_metrics(
                self.metrics['latency'][t],
                self.metrics['jitter'][t],
                self.metrics['energy'][t]
            )
        
        return migration_model.get_metrics()
    
    def get_simulation_results(self):
        """Возврат результатов симуляции"""
        return self.metrics

def run_test_scenario(scenario_type, models):
    """
    Запуск тестирования моделей на заданном сценарии
    
    Параметры:
    scenario_type - тип сценария ('critical', 'standard', 'mixed', 'dynamic')
    models - словарь с моделями миграции
    """
    simulator = NetworkSimulator(scenario_type)
    results = {}
    
    for model_name, model in models.items():
        print(f"Running {model_name} on {scenario_type} scenario...")
        model_results = simulator.run_simulation(model)
        results[model_name] = model_results
    
    # Сохранение результатов в CSV
    save_results_to_csv(scenario_type, results)
    
    return results

def save_results_to_csv(scenario_type, results):
    """Сохранение результатов в CSV-файлы"""
    # Создание DataFrame для каждой метрики
    for metric in ['latency', 'jitter', 'energy']:
        data = {}
        for model_name, model_results in results.items():
            data[model_name] = model_results[metric]
        
        df = pd.DataFrame(data)
        df.to_csv(f"{scenario_type}_{metric}.csv", index=False)
    
    # Сохранение информации о миграциях
    migration_data = {}
    for model_name, model_results in results.items():
        if 'migrations' in model_results:
            migration_data[f"{model_name}_migrations"] = model_results['migrations']
        if 'reactive_migrations' in model_results:
            migration_data[f"{model_name}_reactive"] = model_results['reactive_migrations']
        if 'proactive_migrations' in model_results:
            migration_data[f"{model_name}_proactive"] = model_results['proactive_migrations']
        if 'total_migrations' in model_results:
            migration_data[f"{model_name}_total"] = model_results['total_migrations']
    
    if migration_data:
        pd.DataFrame(migration_data, index=[0]).to_csv(f"{scenario_type}_migrations.csv", index=False)

def run_all_scenarios(models):
    """Запуск всех сценариев тестирования"""
    scenarios = ['critical', 'standard', 'mixed', 'dynamic']
    all_results = {}
    
    for scenario in scenarios:
        all_results[scenario] = run_test_scenario(scenario, models)
    
    return all_results

# Пример использования:
# from threshold_model import ThresholdMarkovModel
# from qlearning_model import QLearningModel
# from hybrid_model import HybridModel
#
# # Инициализация моделей
# nodes = [{'id': i} for i in range(NUM_NODES)]
# services = [{'id': i} for i in range(NUM_SERVICES)]
#
# models = {
#     'threshold': ThresholdMarkovModel(nodes, services),
#     'qlearning': QLearningModel(nodes, services),
#     'hybrid': HybridModel(nodes, services)
# }
#
# # Запуск всех сценариев
# results = run_all_scenarios(models)
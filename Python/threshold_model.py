import numpy as np
import pandas as pd
from collections import defaultdict

class ThresholdModel:
    """
    Модель миграции на основе пороговых значений с марковскими процессами
    """
    def __init__(self, num_nodes=4, num_services=20, threshold=0.75, alpha=0.5, beta=2, 
                 target_load=0.6, history_window=10):
        """
        Инициализация модели
        
        Параметры:
        ----------
        num_nodes : int
            Количество узлов в системе
        num_services : int
            Количество сервисов
        threshold : float
            Пороговое значение загрузки (0.0-1.0)
        alpha : float
            Весовой коэффициент для стоимости миграции
        beta : float
            Коэффициент "жесткости" распределения
        target_load : float
            Целевое значение загрузки узла
        history_window : int
            Размер окна для хранения истории миграций
        """
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.target_load = target_load
        self.history_window = history_window
        
        # Инициализация матрицы вероятностей переходов
        self.transition_matrix = np.ones((num_nodes, num_nodes)) / num_nodes
        np.fill_diagonal(self.transition_matrix, 0)
        self._normalize_matrix()
        
        # Статистика для обновления матрицы переходов
        self.migration_history = []
        self.migration_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        # Метрики эффективности
        self.metrics = {
            'latency': [],
            'jitter': [],
            'energy_consumption': [],
            'migrations_count': 0,
            'failed_migrations': 0
        }
        
        # Текущее состояние системы
        self.node_loads = np.zeros(num_nodes)
        self.service_allocation = np.zeros(num_services, dtype=int)
        self.service_loads = np.zeros(num_services)
        
    def initialize_system(self, node_loads, service_allocation, service_loads):
        """
        Инициализация начального состояния системы
        
        Параметры:
        ----------
        node_loads : numpy.ndarray
            Начальные загрузки узлов (0.0-1.0)
        service_allocation : numpy.ndarray
            Распределение сервисов по узлам
        service_loads : numpy.ndarray
            Загрузка, создаваемая каждым сервисом (0.0-1.0)
        """
        self.node_loads = node_loads.copy()
        self.service_allocation = service_allocation.copy()
        self.service_loads = service_loads.copy()
    
    def _normalize_matrix(self):
        """Нормализация матрицы переходов, чтобы сумма по строкам была равна 1"""
        row_sums = self.transition_matrix.sum(axis=1)
        for i in range(self.num_nodes):
            if row_sums[i] > 0:
                self.transition_matrix[i, :] /= row_sums[i]
    
    def update_transition_matrix(self, source_node, target_node, success):
        """
        Обновление матрицы вероятностей переходов
        
        Параметры:
        ----------
        source_node : int
            Исходный узел
        target_node : int
            Целевой узел
        success : bool
            Флаг успешности миграции
        """
        # Обновление статистики
        key = (source_node, target_node)
        self.migration_stats[key]['total'] += 1
        if success:
            self.migration_stats[key]['successful'] += 1
        
        # Обновление истории
        self.migration_history.append((source_node, target_node, success))
        if len(self.migration_history) > self.history_window:
            self.migration_history.pop(0)
        
        # Пересчет матрицы переходов
        learning_rate = 0.1
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    self.transition_matrix[i, j] = 0
                    continue
                
                key = (i, j)
                if key in self.migration_stats and self.migration_stats[key]['total'] > 0:
                    success_rate = self.migration_stats[key]['successful'] / self.migration_stats[key]['total']
                    # Обновление с учетом скорости обучения
                    self.transition_matrix[i, j] = (1 - learning_rate) * self.transition_matrix[i, j] + \
                                                  learning_rate * success_rate
        
        self._normalize_matrix()
    
    def select_service_for_migration(self, node_index):
        """
        Выбор сервиса для миграции с перегруженного узла
        
        Параметры:
        ----------
        node_index : int
            Индекс перегруженного узла
            
        Возвращает:
        -----------
        int : Индекс выбранного сервиса
        """
        # Получаем сервисы на данном узле
        services_on_node = np.where(self.service_allocation == node_index)[0]
        
        if len(services_on_node) == 0:
            return None
        
        # Выбираем сервис с наибольшей загрузкой
        service_loads_on_node = self.service_loads[services_on_node]
        selected_service_idx = services_on_node[np.argmax(service_loads_on_node)]
        
        return selected_service_idx
    
    def select_target_node(self, source_node, service_load):
        """
        Выбор целевого узла для миграции на основе матрицы вероятностей
        
        Параметры:
        ----------
        source_node : int
            Исходный узел
        service_load : float
            Загрузка, создаваемая мигрирующим сервисом
            
        Возвращает:
        -----------
        int : Индекс выбранного целевого узла
        """
        # Получаем вероятности переходов из исходного узла
        transition_probs = self.transition_matrix[source_node].copy()
        
        # Корректируем вероятности с учетом текущих загрузок узлов
        for j in range(self.num_nodes):
            if j == source_node:
                transition_probs[j] = 0
                continue
            
            # Прогнозируем новую загрузку целевого узла
            new_load = self.node_loads[j] + service_load
            
            # Если новая загрузка превысит порог, снижаем вероятность
            if new_load > self.threshold:
                transition_probs[j] *= 0.1
            
            # Корректировка на основе близости к целевой загрузке
            load_diff = abs(new_load - self.target_load)
            transition_probs[j] *= np.exp(-self.beta * load_diff)
        
        # Нормализуем вероятности
        if np.sum(transition_probs) > 0:
            transition_probs /= np.sum(transition_probs)
        else:
            # Если все вероятности равны нулю, выбираем наименее загруженный узел
            loads = self.node_loads.copy()
            loads[source_node] = float('inf')  # Исключаем исходный узел
            return np.argmin(loads)
        
        # Выбираем узел на основе распределения вероятностей
        target_node = np.random.choice(range(self.num_nodes), p=transition_probs)
        return target_node
    
    def perform_migration(self, service_index, target_node):
        """
        Выполнение миграции сервиса с корректным учетом энергопотребления
        
        Параметры:
        ----------
        service_index : int
            Индекс сервиса для миграции
        target_node : int
            Индекс целевого узла
            
        Возвращает:
        -----------
        bool : Флаг успешности миграции
        float : Задержка, возникшая при миграции
        """
        source_node = self.service_allocation[service_index]
        service_load = self.service_loads[service_index]
        
        # Вычисляем коэффициент сложности миграции
        migration_cost = service_load * (self.node_loads[source_node] + self.node_loads[target_node]) / 2
        
        # Вероятность успешной миграции (обратно пропорциональна сложности)
        success_prob = max(0.85, 1.0 - migration_cost * 0.3)
        
        # Определяем успешность миграции
        success = np.random.random() < success_prob
        
        if success:
            # Обновляем загрузку узлов
            self.node_loads[source_node] -= service_load
            self.node_loads[target_node] += service_load
            
            # Обновляем распределение сервисов
            self.service_allocation[service_index] = target_node
            
            # Вычисляем задержку миграции - высокая для пороговой модели
            latency = 80 + migration_cost * 200  # мс
            
            # Обновляем метрики задержки и джиттера
            self.metrics['latency'].append(latency)
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            # Вычисляем дополнительное энергопотребление при миграции в кВт·ч
            # Примерно 1.5 кВт·ч на единицу migration_cost
            migration_energy = migration_cost * 1.5
            
            # Обновляем последнее значение энергопотребления, добавляя к базовому
            if len(self.metrics['energy_consumption']) > 0:
                self.metrics['energy_consumption'][-1] += migration_energy
                
            self.metrics['migrations_count'] += 1
        else:
            # Миграция не выполнена
            latency = 20  # минимальная задержка попытки
            self.metrics['latency'].append(latency)
            
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            # Добавляем небольшое энергопотребление за попытку миграции
            failed_migration_energy = migration_cost * 0.3
            if len(self.metrics['energy_consumption']) > 0:
                self.metrics['energy_consumption'][-1] += failed_migration_energy
                
            self.metrics['failed_migrations'] += 1
        
        # Обновляем матрицу переходов
        self.update_transition_matrix(source_node, target_node, success)
        
        return success, latency
    
    def check_threshold(self, node_loads=None):
        """
        Проверка превышения порогового значения загрузки с учетом близости к порогу.
        Для реактивной (threshold) модели миграция инициируется, если значение узла
        достигает 90% от установленного порога, даже если оно ещё не превысило его полностью.
        
        Параметры:
        ----------
        node_loads : numpy.ndarray, optional
            Массив загрузок узлов. Если None, используется текущее состояние узлов.
        
        Возвращает:
        -----------
        tuple : (bool, int)
            Флаг наличия перегрузки и индекс узла с наибольшей загрузкой, либо (-1) если перегрузки нет.
        """
        if node_loads is None:
            node_loads = self.node_loads.copy()
        
        # Определяем порог для реакции: если загрузка достигает 90% от установленного порога
        # reactive_threshold = self.threshold * 0.90
        # hysteresis_margin = 0.05
        overloaded = node_loads  >= self.threshold
        
        if np.any(overloaded):
            # Выбираем узел с наибольшей загрузкой
            overload_index = np.argmax(node_loads)
            return True, overload_index
        else:
            return False, -1

    
    def process_step(self, new_loads=None):
        """
        Обработка одного шага симуляции с корректным учетом энергопотребления
        
        Параметры:
        ----------
        new_loads : numpy.ndarray, optional
            Новые загрузки узлов
            
        Возвращает:
        -----------
        dict : Метрики производительности за данный шаг
        """

        if 'energy_consumption' not in self.metrics:
            self.metrics['energy_consumption'] = []
        if 'latency' not in self.metrics:
            self.metrics['latency'] = []
        if 'jitter' not in self.metrics:
            self.metrics['jitter'] = []

        if new_loads is not None:
            self.node_loads = new_loads.copy()
        
        # Всегда добавляем базовое энергопотребление
        baseline_power = 0.05  # кВт базовая мощность на узел
        load_factor = sum(self.node_loads) / self.num_nodes  # средняя загрузка
        baseline_energy = baseline_power * self.num_nodes * load_factor * (5/60)  # кВт·ч за такт
        self.metrics['energy_consumption'].append(baseline_energy)
        
        # Всегда добавляем базовую задержку
        base_latency = 5.0  # мс

        # Проверяем превышение порога
        threshold_exceeded, overloaded_node = self.check_threshold()
        
        # Инициализация метрик шага
        step_metrics = {
            'latency': 0,
            'jitter': 0,
            'migration_performed': False,
            'migration_success': False,
            'migration_mode': 'reactive'
        }
        
        # Расчет базового энергопотребления
        # Для пороговой модели нужно больше ресурсов на постоянный мониторинг
        baseline_power = 0.05  # кВт базовая мощность на узел
        load_factor = sum(self.node_loads) / self.num_nodes  # средняя загрузка
        
        # Пересчитываем в кВт·ч за один такт (5 минут = 1/12 часа)
        baseline_energy = baseline_power * self.num_nodes * load_factor * (5/60)
        
        # Добавляем базовое энергопотребление за этот шаг
        self.metrics['energy_consumption'].append(baseline_energy)
        
        # Базовая задержка обработки запросов (даже без миграций)
        base_latency = 5.0  # мс
        
        if threshold_exceeded:
            # Выбираем сервис для миграции
            service_index = self.select_service_for_migration(overloaded_node)
            
            if service_index is not None:
                # Выбираем целевой узел
                target_node = self.select_target_node(overloaded_node, self.service_loads[service_index])
                
                # Проверяем, что миграция не происходит на тот же узел
                if target_node != overloaded_node:
                    # Выполняем миграцию
                    success, latency = self.perform_migration(service_index, target_node)
                    
                    step_metrics['latency'] = latency
                    step_metrics['migration_performed'] = True
                    step_metrics['migration_success'] = success
                    
                    if len(self.metrics['jitter']) > 0:
                        step_metrics['jitter'] = self.metrics['jitter'][-1]
                else:
                    # Если целевой узел тот же самый, добавляем базовые метрики
                    self.metrics['latency'].append(base_latency)
                    
                    if len(self.metrics['latency']) > 1:
                        jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                        self.metrics['jitter'].append(jitter)
                    else:
                        self.metrics['jitter'].append(0)
            else:
                # Если подходящий сервис не найден, добавляем базовые метрики
                self.metrics['latency'].append(base_latency)
                
                if len(self.metrics['latency']) > 1:
                    jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                    self.metrics['jitter'].append(jitter)
                else:
                    self.metrics['jitter'].append(0)
        else:
            # Если миграция не требуется, добавляем базовые метрики
            self.metrics['latency'].append(base_latency)
            
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
        
        return step_metrics

    
    def get_metrics(self):
        """
        Получение сводных метрик модели
        
        Возвращает:
        -----------
        dict : Сводные метрики производительности
        """
        avg_latency = np.mean(self.metrics['latency']) if len(self.metrics['latency']) > 0 else 0
        avg_jitter = np.mean(self.metrics['jitter']) if len(self.metrics['jitter']) > 0 else 0
        avg_energy = np.mean(self.metrics['energy_consumption']) if len(self.metrics['energy_consumption']) > 0 else 0
        
        return {
            'avg_latency': avg_latency,
            'avg_jitter': avg_jitter,
            'avg_energy_consumption': avg_energy,
            'migrations_count': self.metrics['migrations_count'],
            'failed_migrations': self.metrics['failed_migrations'],
            'success_rate': 1.0 - (self.metrics['failed_migrations'] / 
                                  max(1, self.metrics['migrations_count'] + self.metrics['failed_migrations']))
        }
import numpy as np
import pandas as pd

class ThresholdModel:
    """
    Модель миграции на основе пороговых значений с марковскими процессами.
    
    Параметры:
    ----------
    cpu_threshold : float
        Пороговое значение загрузки CPU (0-1)
    ram_threshold : float
        Пороговое значение загрузки RAM (0-1)
    num_nodes : int
        Количество вычислительных узлов в системе
    num_services : int
        Количество сервисов в системе
    
    Атрибуты:
    ---------
    transition_matrix : numpy.ndarray
        Матрица переходных вероятностей между узлами
    service_placement : dict
        Текущее размещение сервисов по узлам
    metrics_history : list
        История метрик для анализа и визуализации
    """
    
    def __init__(self, cpu_threshold=0.8, ram_threshold=0.8, num_nodes=4, num_services=20):
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.num_nodes = num_nodes
        self.num_services = num_services
        
        # Инициализация матрицы переходных вероятностей
        # Изначально равномерное распределение между всеми узлами, кроме исходного
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Начальное размещение сервисов на узлах (равномерное распределение)
        self.service_placement = self._initialize_service_placement()
        
        # История метрик для анализа производительности
        self.metrics_history = []
        
    def _initialize_transition_matrix(self):
        """Инициализация матрицы переходных вероятностей для марковского процесса"""
        matrix = np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)
        
        # Нормализация по строкам для получения вероятностей
        row_sums = matrix.sum(axis=1)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        
        return normalized_matrix
    
    def _initialize_service_placement(self):
        """Начальное размещение сервисов на узлах"""
        placement = {}
        # Равномерное распределение сервисов по узлам
        for service_id in range(self.num_services):
            node_id = service_id % self.num_nodes
            placement[service_id] = node_id
        return placement
    
    def check_thresholds(self, node_metrics):
        """
        Проверка превышения пороговых значений для узлов
        
        Параметры:
        ----------
        node_metrics : dict
            Словарь с метриками для каждого узла {node_id: {'cpu': float, 'ram': float}}
            
        Возвращает:
        ----------
        list
            Список узлов, превысивших пороговые значения
        """
        overloaded_nodes = []
        
        for node_id, metrics in node_metrics.items():
            if metrics['cpu'] > self.cpu_threshold or metrics['ram'] > self.ram_threshold:
                overloaded_nodes.append(node_id)
                
        return overloaded_nodes
    
    def select_service_for_migration(self, node_id, node_metrics, service_metrics):
        """
        Выбор сервиса для миграции с перегруженного узла
        
        Параметры:
        ----------
        node_id : int
            Идентификатор перегруженного узла
        node_metrics : dict
            Метрики узлов
        service_metrics : dict
            Метрики сервисов
            
        Возвращает:
        ----------
        int или None
            Идентификатор сервиса для миграции или None, если нет подходящих сервисов
        """
        # Получаем сервисы, размещенные на данном узле
        services_on_node = [s_id for s_id, n_id in self.service_placement.items() if n_id == node_id]
        
        if not services_on_node:
            return None
        
        # Выбираем сервис с наибольшим потреблением ресурсов и наименьшим приоритетом
        selected_service = None
        max_resource_usage = -1
        
        for service_id in services_on_node:
            # Комбинированная метрика ресурсопотребления с учетом приоритета
            resource_usage = (service_metrics[service_id]['cpu'] + service_metrics[service_id]['ram']) / service_metrics[service_id]['priority']
            
            if resource_usage > max_resource_usage:
                max_resource_usage = resource_usage
                selected_service = service_id
                
        return selected_service
    
    def select_target_node(self, source_node, service_id, node_metrics, service_metrics):
        """
        Выбор целевого узла для миграции сервиса на основе матрицы переходных вероятностей
        
        Параметры:
        ----------
        source_node : int
            Идентификатор исходного узла
        service_id : int
            Идентификатор мигрируемого сервиса
        node_metrics : dict
            Метрики узлов
        service_metrics : dict
            Метрики сервисов
            
        Возвращает:
        ----------
        int или None
            Идентификатор целевого узла или None, если нет подходящих узлов
        """
        # Получаем вероятности переходов из текущего узла
        transition_probabilities = self.transition_matrix[source_node].copy()
        
        # Исключаем перегруженные узлы и исходный узел
        for node_id, metrics in node_metrics.items():
            # Проверяем, хватит ли ресурсов на целевом узле для размещения сервиса
            if node_id == source_node or \
               metrics['cpu'] + service_metrics[service_id]['cpu'] > self.cpu_threshold or \
               metrics['ram'] + service_metrics[service_id]['ram'] > self.ram_threshold:
                transition_probabilities[node_id] = 0
        
        # Если нет подходящих узлов, возвращаем None
        if np.sum(transition_probabilities) == 0:
            return None
        
        # Нормализуем вероятности
        transition_probabilities = transition_probabilities / np.sum(transition_probabilities)
        
        # Выбираем целевой узел на основе вероятностей
        target_node = np.random.choice(self.num_nodes, p=transition_probabilities)
        
        return target_node
    
    def update_transition_matrix(self, source_node, target_node, success):
        """
        Обновление матрицы переходных вероятностей на основе результата миграции
        
        Параметры:
        ----------
        source_node : int
            Исходный узел
        target_node : int
            Целевой узел
        success : bool
            Успешность миграции
        """
        # Коэффициент обучения
        learning_rate = 0.1
        
        if success:
            # Увеличиваем вероятность перехода к успешному узлу
            self.transition_matrix[source_node, target_node] += learning_rate
        else:
            # Уменьшаем вероятность перехода к неуспешному узлу
            self.transition_matrix[source_node, target_node] = max(0, self.transition_matrix[source_node, target_node] - learning_rate)
        
        # Нормализуем строку матрицы
        row_sum = np.sum(self.transition_matrix[source_node])
        if row_sum > 0:
            self.transition_matrix[source_node] = self.transition_matrix[source_node] / row_sum
    
    def migrate_service(self, service_id, target_node):
        """
        Выполнение миграции сервиса
        
        Параметры:
        ----------
        service_id : int
            Идентификатор сервиса
        target_node : int
            Целевой узел
            
        Возвращает:
        ----------
        bool
            Успешность миграции
        """
        # В реальной системе здесь бы происходила фактическая миграция
        # В нашей модели просто обновляем информацию о размещении
        self.service_placement[service_id] = target_node
        
        return True
    
    def step(self, node_metrics, service_metrics):
        """
        Выполнение одного шага моделирования
        
        Параметры:
        ----------
        node_metrics : dict
            Текущие метрики узлов
        service_metrics : dict
            Текущие метрики сервисов
            
        Возвращает:
        ----------
        dict
            Результаты шага моделирования
        """
        step_results = {
            'migrations': [],
            'overloaded_nodes': []
        }
        
        # Проверяем превышение пороговых значений
        overloaded_nodes = self.check_thresholds(node_metrics)
        step_results['overloaded_nodes'] = overloaded_nodes
        
        # Для каждого перегруженного узла выполняем миграцию
        for node_id in overloaded_nodes:
            # Выбираем сервис для миграции
            service_id = self.select_service_for_migration(node_id, node_metrics, service_metrics)
            
            if service_id is None:
                continue
                
            # Выбираем целевой узел
            target_node = self.select_target_node(node_id, service_id, node_metrics, service_metrics)
            
            if target_node is None:
                continue
                
            # Выполняем миграцию
            success = self.migrate_service(service_id, target_node)
            
            # Обновляем матрицу переходов
            self.update_transition_matrix(node_id, target_node, success)
            
            # Записываем информацию о миграции
            if success:
                step_results['migrations'].append({
                    'service_id': service_id,
                    'source_node': node_id,
                    'target_node': target_node
                })
        
        # Сохраняем метрики для анализа
        self.metrics_history.append({
            'node_metrics': node_metrics.copy(),
            'migrations': len(step_results['migrations']),
            'overloaded_nodes': len(overloaded_nodes)
        })
        
        return step_results
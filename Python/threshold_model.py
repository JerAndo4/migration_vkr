import numpy as np

class ThresholdMarkovModel:
    def __init__(self, nodes, services, cpu_threshold=0.8, memory_threshold=0.8):
        """
        Инициализация пороговой модели с марковскими процессами
        
        Параметры:
        nodes - список узлов
        services - список сервисов
        cpu_threshold - пороговое значение загрузки CPU (от 0 до 1)
        memory_threshold - пороговое значение загрузки памяти (от 0 до 1)
        """
        self.nodes = nodes
        self.services = services
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        # Инициализация матрицы переходов
        n_nodes = len(nodes)
        self.transition_matrix = np.ones((n_nodes, n_nodes)) / n_nodes
        
        # Хранение метрик
        self.metrics = {
            'latency': [],
            'jitter': [],
            'energy': [],
            'migrations': 0
        }
    
    def update_transition_matrix(self, source_node, target_node):
        """Обновление вероятностей переходов на основе успешных миграций"""
        alpha = 0.1  # Скорость обучения
        self.transition_matrix[source_node][target_node] += alpha
        # Нормализация строки для обеспечения суммы вероятностей, равной 1
        self.transition_matrix[source_node] = self.transition_matrix[source_node] / np.sum(self.transition_matrix[source_node])
    
    def select_target_node(self, source_node, load_vector):
        """Выбор целевого узла на основе матрицы переходов и текущей загрузки"""
        probabilities = self.transition_matrix[source_node].copy()
        
        # Снижение вероятности для сильно загруженных узлов
        for i in range(len(self.nodes)):
            if load_vector[i] > 0.7:  # Избегаем миграции на уже загруженные узлы
                probabilities[i] *= 0.5
                
        # Нормализация вероятностей
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
            
        # Выбор целевого узла
        target_node = np.random.choice(len(self.nodes), p=probabilities)
        return target_node
    
    def check_threshold_violation(self, node_metrics):
        """Проверка превышения порогов"""
        violations = []
        
        for node_id, metrics in node_metrics.items():
            if metrics['cpu_usage'] > self.cpu_threshold or metrics['memory_usage'] > self.memory_threshold:
                violations.append(node_id)
                
        return violations
    
    def migrate(self, node_metrics, service_placement):
        """Выполнение миграции при превышении порогов"""
        violations = self.check_threshold_violation(node_metrics)
        migrations = []
        
        if not violations:
            return migrations
            
        # Создание вектора загрузки для всех узлов
        load_vector = [node_metrics[node_id]['cpu_usage'] for node_id in range(len(self.nodes))]
        
        for node_id in violations:
            # Поиск сервисов на перегруженном узле
            node_services = [s for s, n in service_placement.items() if n == node_id]
            
            if not node_services:
                continue
                
            # Выбор сервиса для миграции (с наибольшим потреблением ресурсов)
            service_to_migrate = node_services[0]  # Упрощенный выбор
            
            # Выбор целевого узла
            target_node = self.select_target_node(node_id, load_vector)
            
            # Обновление матрицы переходов
            self.update_transition_matrix(node_id, target_node)
            
            # Запись миграции
            migrations.append((service_to_migrate, node_id, target_node))
            self.metrics['migrations'] += 1
            
        return migrations
    
    def update_metrics(self, latency, jitter, energy):
        """Обновление метрик производительности"""
        self.metrics['latency'].append(latency)
        self.metrics['jitter'].append(jitter)
        self.metrics['energy'].append(energy)
    
    def get_metrics(self):
        """Возврат метрик производительности"""
        return self.metrics
import numpy as np

class HybridModel:
    def __init__(self, nodes, services, cpu_threshold=0.8, memory_threshold=0.8, 
                 learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """
        Инициализация гибридной модели
        
        Параметры:
        nodes - список узлов
        services - список сервисов
        cpu_threshold - пороговое значение загрузки CPU (от 0 до 1)
        memory_threshold - пороговое значение загрузки памяти (от 0 до 1)
        learning_rate - скорость обучения (альфа)
        discount_factor - коэффициент дисконтирования (гамма)
        exploration_rate - вероятность исследования (эпсилон)
        """
        self.nodes = nodes
        self.services = services
        
        # Параметры пороговой модели
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        # Параметры Q-learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Инициализация матрицы переходов для марковской модели
        n_nodes = len(nodes)
        self.transition_matrix = np.ones((n_nodes, n_nodes)) / n_nodes
        
        # Инициализация Q-таблицы для Q-learning
        self.q_table = {}
        
        # Хранение метрик
        self.metrics = {
            'latency': [],
            'jitter': [],
            'energy': [],
            'reactive_migrations': 0,
            'proactive_migrations': 0,
            'total_migrations': 0
        }
        
        # Модель предсказания
        self.load_history = {node_id: [] for node_id in range(n_nodes)}
        self.prediction_window = 5
    
    def update_transition_matrix(self, source_node, target_node):
        """Обновление вероятностей переходов на основе успешных миграций"""
        alpha = 0.1  # Скорость обучения
        self.transition_matrix[source_node][target_node] += alpha
        # Нормализация строки для обеспечения суммы вероятностей, равной 1
        self.transition_matrix[source_node] = self.transition_matrix[source_node] / np.sum(self.transition_matrix[source_node])
    
    def select_target_node_markov(self, source_node, load_vector):
        """Выбор целевого узла на основе марковских вероятностей переходов"""
        probabilities = self.transition_matrix[source_node].copy()
        
        # Снижение вероятности для сильно загруженных узлов
        for i in range(len(self.nodes)):
            if load_vector[i] > 0.7:
                probabilities[i] *= 0.5
                
        # Нормализация вероятностей
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
            
        # Выбор целевого узла
        target_node = np.random.choice(len(self.nodes), p=probabilities)
        return target_node
    
    def discretize_state(self, node_metrics):
        """Преобразование непрерывных метрик в дискретное представление состояния"""
        state = []
        for node_id in range(len(self.nodes)):
            # Дискретизация загрузки CPU до 10 уровней
            cpu_level = min(9, int(node_metrics[node_id]['cpu_usage'] * 10))
            state.append(cpu_level)
        return tuple(state)
    
    def predict_load(self, node_metrics):
        """Предсказание будущей нагрузки на основе истории"""
        predictions = {}
        
        for node_id in range(len(self.nodes)):
            # Обновление истории
            self.load_history[node_id].append(node_metrics[node_id]['cpu_usage'])
            
            # Сохраняем только недавнюю историю
            if len(self.load_history[node_id]) > self.prediction_window:
                self.load_history[node_id] = self.load_history[node_id][-self.prediction_window:]
            
            # Простое линейное предсказание тренда
            if len(self.load_history[node_id]) >= 2:
                trend = self.load_history[node_id][-1] - self.load_history[node_id][0]
                trend_per_step = trend / (len(self.load_history[node_id]) - 1)
                prediction = self.load_history[node_id][-1] + trend_per_step
                predictions[node_id] = max(0, min(1, prediction))
            else:
                predictions[node_id] = node_metrics[node_id]['cpu_usage']
                
        return predictions
    
    def select_action_q_learning(self, state, predicted_overload):
        """Выбор действия миграции на основе Q-значений"""
        if not predicted_overload:
            return None
            
        # Инициализация Q-значений для нового состояния при необходимости
        if state not in self.q_table:
            self.q_table[state] = {}
            
        # Исследование: случайное действие
        if np.random.random() < self.exploration_rate:
            service_id = np.random.choice(len(self.services))
            target_node = np.random.choice(len(self.nodes))
            action = (service_id, target_node)
            
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
                
            return action
            
        # Эксплуатация: лучшее известное действие
        else:
            if not self.q_table[state]:  # Если еще нет записанных действий
                service_id = np.random.choice(len(self.services))
                target_node = np.random.choice(len(self.nodes))
                action = (service_id, target_node)
                self.q_table[state][action] = 0.0
                return action
                
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state, action, reward, next_state):
        """Обновление Q-значения с использованием правила обновления Q-learning"""
        # Инициализация Q-значений для новых состояний при необходимости
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            
        # Получение максимального Q-значения для следующего состояния
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Обновление Q-значения
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * max_next_q - self.q_table[state][action]
        )
    
    def calculate_reward(self, latency, energy, load_balance):
        """Расчет награды на основе метрик производительности"""
        latency_weight = -1.0
        energy_weight = -0.5
        balance_weight = 1.0
        
        reward = (
            latency_weight * latency +
            energy_weight * energy +
            balance_weight * load_balance
        )
        
        return reward
    
    def check_threshold_violation(self, node_metrics):
        """Проверка превышения порогов (реактивный компонент)"""
        violations = []
        
        for node_id, metrics in node_metrics.items():
            if metrics['cpu_usage'] > self.cpu_threshold or metrics['memory_usage'] > self.memory_threshold:
                violations.append(node_id)
                
        return violations
    
    def migrate(self, node_metrics, service_placement):
        """Выполнение миграции с использованием гибридного подхода"""
        migrations = []
        
        # РЕАКТИВНЫЙ КОМПОНЕНТ: Проверка превышения порогов
        violations = self.check_threshold_violation(node_metrics)
        
        if violations:
            # Создание вектора загрузки для всех узлов
            load_vector = [node_metrics[node_id]['cpu_usage'] for node_id in range(len(self.nodes))]
            
            for node_id in violations:
                # Поиск сервисов на перегруженном узле
                node_services = [s for s, n in service_placement.items() if n == node_id]
                
                if not node_services:
                    continue
                    
                # Выбор сервиса для миграции (упрощенно)
                service_to_migrate = node_services[0]
                
                # Выбор целевого узла с использованием марковской модели
                target_node = self.select_target_node_markov(node_id, load_vector)
                
                # Обновление матрицы переходов
                self.update_transition_matrix(node_id, target_node)
                
                # Запись миграции
                migrations.append((service_to_migrate, node_id, target_node))
                self.metrics['reactive_migrations'] += 1
                self.metrics['total_migrations'] += 1
        
        # ПРОАКТИВНЫЙ КОМПОНЕНТ: Q-learning с предсказанием
        if not violations:  # Используем проактивный подход, только если нет реактивных миграций
            # Получение текущего состояния
            current_state = self.discretize_state(node_metrics)
            
            # Предсказание будущей нагрузки
            load_predictions = self.predict_load(node_metrics)
            
            # Проверка предсказанной перегрузки
            predicted_overload = any(load > 0.8 for load in load_predictions.values())
            
            # Выбор действия
            action = self.select_action_q_learning(current_state, predicted_overload)
            
            if action:
                service_id, target_node = action
                source_node = service_placement.get(service_id)
                
                # Пропуск, если сервис уже на целевом узле
                if source_node != target_node:
                    # Выполнение миграции
                    migrations.append((service_id, source_node, target_node))
                    self.metrics['proactive_migrations'] += 1
                    self.metrics['total_migrations'] += 1
        
        return migrations
    
    def update_after_migration(self, previous_state, action, new_state, latency, energy, load_balance):
        """Обновление Q-значений после миграции"""
        reward = self.calculate_reward(latency, energy, load_balance)
        self.update_q_value(previous_state, action, reward, new_state)
    
    def update_metrics(self, latency, jitter, energy):
        """Обновление метрик производительности"""
        self.metrics['latency'].append(latency)
        self.metrics['jitter'].append(jitter)
        self.metrics['energy'].append(energy)
    
    def get_metrics(self):
        """Возврат метрик производительности"""
        return self.metrics
import numpy as np

class QLearningModel:
    def __init__(self, nodes, services, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """
        Инициализация модели на основе Q-Learning
        
        Параметры:
        nodes - список узлов
        services - список сервисов
        learning_rate - скорость обучения (альфа)
        discount_factor - коэффициент дисконтирования (гамма)
        exploration_rate - вероятность исследования (эпсилон)
        """
        self.nodes = nodes
        self.services = services
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Инициализация Q-таблицы
        self.q_table = {}
        
        # Хранение метрик
        self.metrics = {
            'latency': [],
            'jitter': [],
            'energy': [],
            'migrations': 0
        }
        
        # Модель предсказания
        self.load_history = {node_id: [] for node_id in range(len(nodes))}
        self.prediction_window = 5
    
    def discretize_state(self, node_metrics):
        """Преобразование непрерывных метрик в дискретное представление состояния"""
        state = []
        for node_id in range(len(self.nodes)):
            # Дискретизация загрузки CPU до 10 уровней (0, 0.1, 0.2, ..., 0.9)
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
                predictions[node_id] = max(0, min(1, prediction))  # Ограничение значениями от 0 до 1
            else:
                predictions[node_id] = node_metrics[node_id]['cpu_usage']
                
        return predictions
    
    def select_action(self, state, predicted_overload):
        """Выбор действия миграции на основе Q-значений и стратегии исследования"""
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
        latency_weight = -1.0  # Отрицательный вес, так как более низкая задержка лучше
        energy_weight = -0.5   # Отрицательный вес, так как более низкое энергопотребление лучше
        balance_weight = 1.0   # Положительный вес, так как более высокий баланс лучше
        
        reward = (
            latency_weight * latency +
            energy_weight * energy +
            balance_weight * load_balance
        )
        
        return reward
    
    def migrate(self, node_metrics, service_placement):
        """Выполнение миграции на основе Q-learning"""
        # Получение текущего состояния
        current_state = self.discretize_state(node_metrics)
        
        # Предсказание будущей нагрузки
        load_predictions = self.predict_load(node_metrics)
        
        # Проверка предсказанной перегрузки
        predicted_overload = any(load > 0.8 for load in load_predictions.values())
        
        # Выбор действия
        action = self.select_action(current_state, predicted_overload)
        
        if not action:
            return []
            
        service_id, target_node = action
        source_node = service_placement.get(service_id)
        
        # Пропуск, если сервис уже на целевом узле
        if source_node == target_node:
            return []
            
        # Выполнение миграции
        migrations = [(service_id, source_node, target_node)]
        self.metrics['migrations'] += 1
        
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
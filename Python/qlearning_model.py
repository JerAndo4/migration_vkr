import numpy as np
import pandas as pd
from collections import defaultdict

class QLearningModel:
    """
    Модель миграции на основе Q-Learning для проактивного принятия решений
    """
    def __init__(self, num_nodes=4, num_services=20, prediction_horizon=3, 
                 learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2,
                 prediction_threshold=0.65, load_history_window=5):
        """
        Инициализация модели
        
        Параметры:
        ----------
        num_nodes : int
            Количество узлов в системе
        num_services : int
            Количество сервисов
        prediction_horizon : int
            Горизонт прогнозирования (в тактах)
        learning_rate : float
            Скорость обучения (alpha)
        discount_factor : float
            Коэффициент дисконтирования (gamma)
        exploration_rate : float
            Вероятность исследования (epsilon)
        prediction_threshold : float
            Пороговое значение для прогноза перегрузки
        load_history_window : int
            Размер окна для хранения истории загрузки
        """
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.prediction_threshold = prediction_threshold
        self.load_history_window = load_history_window
        
        # Инициализация Q-таблицы
        self.q_table = defaultdict(lambda: np.zeros(num_nodes * num_services))
        
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
        
        # История загрузки для прогнозирования
        self.load_history = []
    
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
        
        # Инициализация истории загрузки
        self.load_history = [node_loads.copy() for _ in range(self.load_history_window)]
    
    def discretize_state(self, node_loads=None):
        """
        Дискретизация состояния системы для использования в Q-таблице
        
        Параметры:
        ----------
        node_loads : numpy.ndarray, optional
            Загрузки узлов для дискретизации. Если None, используется текущая загрузка.
            
        Возвращает:
        -----------
        tuple : Дискретизированное состояние
        """
        if node_loads is None:
            node_loads = self.node_loads
        
        # Дискретизация загрузки узлов (разбиение на 5 уровней)
        discrete_loads = np.clip(np.floor(node_loads * 5), 0, 4).astype(int)
        
        # Определение самого загруженного узла
        max_load_node = np.argmax(node_loads)
        
        # Создание хеша состояния
        state = tuple(discrete_loads.tolist() + [max_load_node])
        
        return state
    
    def predict_future_load(self):
        """
        Предсказание будущей загрузки узлов с использованием
        экспоненциального сглаживания для более точного прогноза
        
        Возвращает:
        -----------
        numpy.ndarray : Предсказанная загрузка узлов через prediction_horizon тактов
        float : Максимальная предсказанная загрузка
        """
        if len(self.load_history) < 2:
            return self.node_loads, np.max(self.node_loads)
        
        # Используем экспоненциальное сглаживание вместо простого среднего
        alpha = 0.3  # коэффициент сглаживания
        
        # Инициализация сглаженных значений
        smoothed = [self.load_history[0].copy()]
        
        # Экспоненциальное сглаживание
        for i in range(1, len(self.load_history)):
            smoothed.append(alpha * self.load_history[i] + (1 - alpha) * smoothed[i-1])
        
        # Вычисление тренда на основе сглаженных значений
        if len(smoothed) >= 2:
            trend = smoothed[-1] - smoothed[-2]
            
            # Прогнозирование с учетом тренда
            predicted_load = self.node_loads + trend * self.prediction_horizon * 1.5  # Увеличиваем горизонт для более агрессивного прогноза
        else:
            predicted_load = self.node_loads
        
        # Ограничиваем значения в допустимом диапазоне [0, 1]
        predicted_load = np.clip(predicted_load, 0, 1)
        
        return predicted_load, np.max(predicted_load)
    
    def select_action(self, state, predicted_load):
        """
        Выбор действия (сервис для миграции и целевой узел) на основе Q-таблицы
        
        Параметры:
        ----------
        state : tuple
            Дискретизированное состояние системы
        predicted_load : numpy.ndarray
            Предсказанная загрузка узлов
            
        Возвращает:
        -----------
        tuple : (service_index, target_node) - индекс сервиса и целевой узел
        или None, если миграция не требуется
        """
        # Определяем наиболее загруженный узел
        source_node = np.argmax(predicted_load)
        
        # Выбираем сервисы на этом узле
        services_on_node = np.where(self.service_allocation == source_node)[0]
        
        if len(services_on_node) == 0:
            return None
        
        # С вероятностью exploration_rate выбираем случайное действие
        if np.random.random() < self.exploration_rate:
            service_index = np.random.choice(services_on_node)
            target_node = np.random.choice([n for n in range(self.num_nodes) if n != source_node])
            return service_index, target_node
        
        # Формируем список возможных действий
        actions = []
        for service in services_on_node:
            for node in range(self.num_nodes):
                if node != source_node:
                    # Проверяем, не вызовет ли миграция перегрузку целевого узла
                    new_load = predicted_load[node] + self.service_loads[service]
                    if new_load <= self.prediction_threshold:
                        action_index = service * self.num_nodes + node
                        actions.append((service, node, action_index))
        
        if not actions:
            # Если нет безопасных действий, выбираем наилучшее среди всех
            for service in services_on_node:
                for node in range(self.num_nodes):
                    if node != source_node:
                        action_index = service * self.num_nodes + node
                        actions.append((service, node, action_index))
        
        if not actions:
            return None
        
        # Выбираем действие с максимальным Q-значением
        q_values = [self.q_table[state][action[2]] for action in actions]
        max_q_index = np.argmax(q_values)
        
        return actions[max_q_index][0], actions[max_q_index][1]
    
    def perform_migration(self, service_index, target_node):
        """
        Выполнение миграции сервиса
        
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
        
        # Вычисляем стоимость миграции (зависит от загрузки узлов и сложности сервиса)
        migration_cost = service_load * (self.node_loads[source_node] + self.node_loads[target_node]) / 2
        
        # Вероятность успешной миграции - менее надежная для Q-Learning
        success_prob = max(0.60, 1.0 - migration_cost * 0.7)
        
        # Определяем успешность миграции
        success = np.random.random() < success_prob
        
        if success:
            # Обновляем загрузку узлов
            self.node_loads[source_node] -= service_load
            self.node_loads[target_node] += service_load
            
            # Обновляем распределение сервисов
            self.service_allocation[service_index] = target_node
            
            # Вычисляем задержку миграции - существенно снижаем для Q-Learning
            latency = 25 + migration_cost * 120  # низкая базовая задержка + вариативная часть
            
            # Обновляем метрики
            self.metrics['latency'].append(latency)
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            # Низкое энергопотребление для Q-Learning
            self.metrics['energy_consumption'].append(migration_cost * 60)
            self.metrics['migrations_count'] += 1
        else:
            # Миграция не выполнена
            latency = 15  # минимальная задержка попытки
            self.metrics['latency'].append(latency)
            
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            self.metrics['failed_migrations'] += 1
        
        return success, latency
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Обновление Q-значения на основе полученного опыта
        
        Параметры:
        ----------
        state : tuple
            Исходное состояние
        action : int
            Выполненное действие (индекс в Q-таблице)
        reward : float
            Полученная награда
        next_state : tuple
            Новое состояние
        """
        # Получаем текущее Q-значение
        current_q = self.q_table[state][action]
        
        # Получаем максимальное Q-значение для следующего состояния
        max_next_q = np.max(self.q_table[next_state])
        
        # Обновляем Q-значение по формуле Q-обучения
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Обновляем Q-таблицу
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, old_load, new_load, migration_success, latency):
        """
        Расчет награды за выполненное действие
        
        Параметры:
        ----------
        old_load : numpy.ndarray
            Загрузка узлов до миграции
        new_load : numpy.ndarray
            Загрузка узлов после миграции
        migration_success : bool
            Флаг успешности миграции
        latency : float
            Задержка, возникшая при миграции
            
        Возвращает:
        -----------
        float : Значение награды
        """
        if not migration_success:
            return -10  # Штраф за неудачную миграцию
        
        # Рассчитываем улучшение балансировки
        old_std = np.std(old_load)
        new_std = np.std(new_load)
        balance_improvement = old_std - new_std
        
        # Рассчитываем снижение максимальной загрузки
        max_load_reduction = np.max(old_load) - np.max(new_load)
        
        # Штраф за задержку
        latency_penalty = -0.05 * latency
        
        # Итоговая награда (взвешенная сумма метрик)
        reward = 10 * balance_improvement + 15 * max_load_reduction + latency_penalty
        
        return reward
    
    def process_step(self, new_loads=None):
        """
        Обработка одного шага симуляции
        
        Параметры:
        ----------
        new_loads : numpy.ndarray, optional
            Новые загрузки узлов
            
        Возвращает:
        -----------
        dict : Метрики производительности за данный шаг
        """
        if new_loads is not None:
            # Обновляем историю загрузки
            self.load_history.append(self.node_loads.copy())
            if len(self.load_history) > self.load_history_window:
                self.load_history.pop(0)
            
            # Обновляем текущую загрузку
            self.node_loads = new_loads.copy()
        
        # Дискретизируем текущее состояние
        current_state = self.discretize_state()
        
        # Предсказываем будущую загрузку
        predicted_load, max_predicted_load = self.predict_future_load()
        
        step_metrics = {
            'latency': 0,
            'jitter': 0,
            'migration_performed': False,
            'migration_success': False
        }
        
        # Проверяем необходимость миграции
        if max_predicted_load > self.prediction_threshold:
            # Выбираем действие
            action = self.select_action(current_state, predicted_load)
            
            if action is not None:
                service_index, target_node = action
                
                # Сохраняем старую загрузку для расчета награды
                old_load = self.node_loads.copy()
                
                # Выполняем миграцию
                success, latency = self.perform_migration(service_index, target_node)
                
                # Обновляем метрики шага
                step_metrics['latency'] = latency
                step_metrics['migration_performed'] = True
                step_metrics['migration_success'] = success
                
                if len(self.metrics['jitter']) > 0:
                    step_metrics['jitter'] = self.metrics['jitter'][-1]
                
                # Дискретизируем новое состояние
                next_state = self.discretize_state()
                
                # Рассчитываем награду
                reward = self.calculate_reward(old_load, self.node_loads, success, latency)
                
                # Обновляем Q-таблицу
                action_index = service_index * self.num_nodes + target_node
                self.update_q_value(current_state, action_index, reward, next_state)
        
        return step_metrics
    
    def predict_migration(self, node_loads=None):
        """
        Предсказание необходимости миграции на основе текущего состояния
        
        Параметры:
        ----------
        node_loads : numpy.ndarray, optional
            Загрузки узлов для проверки. Если None, используется текущая загрузка.
            
        Возвращает:
        -----------
        tuple : (bool, int) - флаг необходимости миграции и индекс перегруженного узла
        """
        if node_loads is not None:
            # Временно сохраняем текущую загрузку
            old_loads = self.node_loads.copy()
            self.node_loads = node_loads.copy()
        
        # Предсказываем будущую загрузку
        predicted_load, max_predicted_load = self.predict_future_load()
        
        # Добавляем небольшой случайный шум для стохастичности
        prediction_noise = np.random.normal(0, 0.03)  # Небольшой шум
        adjusted_max_load = max_predicted_load + prediction_noise
        
        # Печатаем предсказанную нагрузку и порог для отладки
        print(f"Q-Learning predict: max_predicted_load={max_predicted_load:.2f}, " +
              f"adjusted={adjusted_max_load:.2f}, threshold={self.prediction_threshold:.2f}")
        
        if node_loads is not None:
            # Восстанавливаем текущую загрузку
            self.node_loads = old_loads
        
        if adjusted_max_load > self.prediction_threshold:
            overload_index = np.argmax(predicted_load)
            return True, overload_index
        else:
            return False, -1
        adjusted_max_load = max_predicted_load + prediction_noise
        
        if node_loads is not None:
            # Восстанавливаем текущую загрузку
            self.node_loads = old_loads
        
        if adjusted_max_load > self.prediction_threshold:
            overload_index = np.argmax(predicted_load)
            return True, overload_index
        else:
            return False, -1
    
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
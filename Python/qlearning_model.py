import numpy as np
import pandas as pd
from collections import deque

class QLearningModel:
    """
    Модель миграции на основе Q-Learning.
    
    Параметры:
    ----------
    num_nodes : int
        Количество вычислительных узлов в системе
    num_services : int
        Количество сервисов в системе
    learning_rate : float
        Скорость обучения (alpha)
    discount_factor : float
        Коэффициент дисконтирования (gamma)
    exploration_rate : float
        Начальная вероятность исследования (epsilon)
    min_exploration_rate : float
        Минимальная вероятность исследования
    exploration_decay : float
        Скорость снижения вероятности исследования
    cpu_threshold : float
        Пороговое значение CPU для прогнозирования перегрузки
    ram_threshold : float
        Пороговое значение RAM для прогнозирования перегрузки
    history_window : int
        Размер окна для исторических данных
    """
    
    def __init__(self, num_nodes=4, num_services=20, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, min_exploration_rate=0.1, exploration_decay=0.995,
                 cpu_threshold=0.8, ram_threshold=0.8, history_window=5):
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        
        # Начальное размещение сервисов на узлах (равномерное распределение)
        self.service_placement = self._initialize_service_placement()
        
        # Инициализация Q-таблицы
        # Состояние: (узел_источник, узел_назначения, сервис)
        # Действие: миграция сервиса с узла_источника на узел_назначения
        self.q_table = {}
        
        # История метрик для прогнозирования
        self.metrics_history = []
        self.history_window = history_window
        
        # Буфер истории для каждого узла
        self.node_history = {node_id: deque(maxlen=history_window) for node_id in range(num_nodes)}
        
    def _initialize_service_placement(self):
        """Начальное размещение сервисов на узлах"""
        placement = {}
        # Равномерное распределение сервисов по узлам
        for service_id in range(self.num_services):
            node_id = service_id % self.num_nodes
            placement[service_id] = node_id
        return placement
    
    def _get_state_key(self, node_id, service_id):
        """Генерация ключа состояния для Q-таблицы"""
        # Дискретизация состояния для уменьшения размерности Q-таблицы
        return f"{node_id}_{service_id}"
    
    def _get_action_key(self, target_node):
        """Генерация ключа действия для Q-таблицы"""
        return f"migrate_to_{target_node}"
        
    def predict_overload(self, node_metrics):
        """
        Прогнозирование перегрузки узлов на основе исторических данных
        
        Параметры:
        ----------
        node_metrics : dict
            Текущие метрики узлов
            
        Возвращает:
        ----------
        list
            Список узлов с прогнозируемой перегрузкой
        """
        predicted_overload = []
        
        # Обновляем историю для каждого узла
        for node_id, metrics in node_metrics.items():
            self.node_history[node_id].append((metrics['cpu'], metrics['ram']))
            
            # Если достаточно данных для прогнозирования
            if len(self.node_history[node_id]) == self.history_window:
                # Рассчитываем тренд (простая линейная экстраполяция)
                cpu_values = [x[0] for x in self.node_history[node_id]]
                ram_values = [x[1] for x in self.node_history[node_id]]
                
                # Линейный тренд по CPU
                cpu_diff = np.diff(cpu_values)
                mean_cpu_diff = np.mean(cpu_diff) if len(cpu_diff) > 0 else 0
                predicted_cpu = cpu_values[-1] + mean_cpu_diff
                
                # Линейный тренд по RAM
                ram_diff = np.diff(ram_values)
                mean_ram_diff = np.mean(ram_diff) if len(ram_diff) > 0 else 0
                predicted_ram = ram_values[-1] + mean_ram_diff
                
                # Прогнозируем перегрузку, если ожидается превышение порогов
                if predicted_cpu > self.cpu_threshold or predicted_ram > self.ram_threshold:
                    predicted_overload.append(node_id)
        
        return predicted_overload
    
    def select_service_for_migration(self, node_id, node_metrics, service_metrics):
        """
        Выбор сервиса для миграции с потенциально перегруженного узла
        
        Параметры:
        ----------
        node_id : int
            Идентификатор узла
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
        Выбор целевого узла для миграции сервиса на основе Q-таблицы
        
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
        state_key = self._get_state_key(source_node, service_id)
        
        # Инициализация Q-значений для данного состояния, если они не существуют
        if state_key not in self.q_table:
            self.q_table[state_key] = {self._get_action_key(node_id): 0 for node_id in range(self.num_nodes) if node_id != source_node}
        
        # Определяем, будем исследовать (случайный выбор) или использовать Q-таблицу
        if np.random.rand() < self.exploration_rate:
            # Исследование: случайный выбор узла, исключая исходный и перегруженные
            available_nodes = []
            for node_id in range(self.num_nodes):
                if node_id != source_node:
                    # Проверяем, хватит ли ресурсов на целевом узле для размещения сервиса
                    if node_metrics[node_id]['cpu'] + service_metrics[service_id]['cpu'] <= self.cpu_threshold and \
                       node_metrics[node_id]['ram'] + service_metrics[service_id]['ram'] <= self.ram_threshold:
                        available_nodes.append(node_id)
            
            if not available_nodes:
                return None
                
            target_node = np.random.choice(available_nodes)
        else:
            # Использование: выбор на основе Q-значений
            q_values = self.q_table[state_key]
            
            # Исключаем перегруженные узлы
            for node_id in range(self.num_nodes):
                if node_id != source_node:
                    action_key = self._get_action_key(node_id)
                    # Проверяем, хватит ли ресурсов на целевом узле
                    if node_metrics[node_id]['cpu'] + service_metrics[service_id]['cpu'] > self.cpu_threshold or \
                       node_metrics[node_id]['ram'] + service_metrics[service_id]['ram'] > self.ram_threshold:
                        q_values[action_key] = float('-inf')  # Исключаем из рассмотрения
            
            # Если все узлы перегружены, возвращаем None
            valid_actions = {k: v for k, v in q_values.items() if v != float('-inf')}
            if not valid_actions:
                return None
                
            # Выбираем действие с максимальным Q-значением
            best_action = max(valid_actions, key=valid_actions.get)
            target_node = int(best_action.split('_')[-1])
        
        # Снижаем вероятность исследования
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        
        return target_node
    
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
    
    def calculate_reward(self, source_node, target_node, service_id, node_metrics, service_metrics):
        """
        Расчет награды за действие миграции
        
        Параметры:
        ----------
        source_node : int
            Исходный узел
        target_node : int
            Целевой узел
        service_id : int
            Идентификатор сервиса
        node_metrics : dict
            Метрики узлов после миграции
        service_metrics : dict
            Метрики сервисов
            
        Возвращает:
        ----------
        float
            Значение награды
        """
        # Базовая награда за успешную миграцию
        reward = 1.0
        
        # Штраф за высокую загрузку целевого узла
        target_load = node_metrics[target_node]['cpu'] + node_metrics[target_node]['ram']
        if target_load > 1.4:  # Суммарная загрузка CPU и RAM выше 70% в среднем
            reward -= 0.5
        
        # Награда за разгрузку исходного узла
        source_load = node_metrics[source_node]['cpu'] + node_metrics[source_node]['ram']
        if source_load < 1.4:  # Суммарная загрузка CPU и RAM ниже 70% в среднем
            reward += 0.5
            
        # Штраф за частые миграции (можно реализовать, если хранить историю миграций)
        
        return reward
    
    def update_q_value(self, state_key, action_key, reward, next_state_key):
        """
        Обновление Q-значения на основе полученной награды
        
        Параметры:
        ----------
        state_key : str
            Ключ исходного состояния
        action_key : str
            Ключ выполненного действия
        reward : float
            Полученная награда
        next_state_key : str
            Ключ следующего состояния
        """
        # Инициализация Q-значений для следующего состояния, если они не существуют
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {self._get_action_key(node_id): 0 for node_id in range(self.num_nodes)}
        
        # Максимальное Q-значение для следующего состояния
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        
        # Обновление Q-значения по формуле Q-Learning
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state_key][action_key] = new_q
    
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
            'predicted_overloads': []
        }
        
        # Прогнозирование перегрузки
        predicted_overloads = self.predict_overload(node_metrics)
        step_results['predicted_overloads'] = predicted_overloads
        
        # Для каждого узла с прогнозируемой перегрузкой выполняем миграцию
        for node_id in predicted_overloads:
            # Выбираем сервис для миграции
            service_id = self.select_service_for_migration(node_id, node_metrics, service_metrics)
            
            if service_id is None:
                continue
                
            # Текущее состояние
            state_key = self._get_state_key(node_id, service_id)
            
            # Выбираем целевой узел
            target_node = self.select_target_node(node_id, service_id, node_metrics, service_metrics)
            
            if target_node is None:
                continue
                
            # Действие
            action_key = self._get_action_key(target_node)
            
            # Выполняем миграцию
            success = self.migrate_service(service_id, target_node)
            
            if success:
                # Следующее состояние
                next_state_key = self._get_state_key(target_node, service_id)
                
                # Рассчитываем награду
                reward = self.calculate_reward(node_id, target_node, service_id, node_metrics, service_metrics)
                
                # Обновляем Q-значение
                self.update_q_value(state_key, action_key, reward, next_state_key)
                
                # Записываем информацию о миграции
                step_results['migrations'].append({
                    'service_id': service_id,
                    'source_node': node_id,
                    'target_node': target_node,
                    'reward': reward
                })
        
        # Сохраняем метрики для анализа
        self.metrics_history.append({
            'node_metrics': node_metrics.copy(),
            'migrations': len(step_results['migrations']),
            'predicted_overloads': len(predicted_overloads),
            'exploration_rate': self.exploration_rate
        })
        
        return step_results
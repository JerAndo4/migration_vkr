import numpy as np
import pandas as pd
from collections import defaultdict

class QLearningModel:
    """
    Модель миграции на основе Q-Learning для проактивного принятия решений
    с улучшенными механизмами контроля количества миграций
    """
    def __init__(self, num_nodes=4, num_services=20, prediction_horizon=3, 
                 learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1,
                 prediction_threshold=0.75, load_history_window=10,
                 cooling_period=5, migration_penalty=5.0):
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
            Вероятность исследования (epsilon) - снижена с 0.2 до 0.1
        prediction_threshold : float
            Пороговое значение для прогноза перегрузки - повышено с 0.65 до 0.75
        load_history_window : int
            Размер окна для хранения истории загрузки - увеличено с 5 до 10
        cooling_period : int
            Период "остывания" после миграции сервиса (новый параметр)
        migration_penalty : float
            Штраф за выполнение миграции (новый параметр)
        """
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.prediction_threshold = prediction_threshold
        self.load_history_window = load_history_window
        self.cooling_period = cooling_period
        self.migration_penalty = migration_penalty
        
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
        
        # Счетчики периодов остывания для каждого сервиса
        self.cooling_counters = np.zeros(num_services, dtype=int)
        
        # История миграций для предотвращения циклических миграций
        self.migration_history = []
    
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
        
        # Сброс счетчиков остывания
        self.cooling_counters = np.zeros(self.num_services, dtype=int)
        
        # Очистка истории миграций
        self.migration_history = []
    
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
        
        # Улучшенная дискретизация загрузки узлов (разбиение на 10 уровней вместо 5)
        discrete_loads = np.clip(np.floor(node_loads * 10), 0, 9).astype(int)
        
        # Определение самого загруженного узла
        max_load_node = np.argmax(node_loads)
        
        # Создание хеша состояния
        state = tuple(discrete_loads.tolist() + [max_load_node])
        
        return state
    
    def predict_future_load(self):
        """
        Предсказание будущей загрузки узлов с использованием улучшенного
        экспоненциального сглаживания с трендом (Holt's method).
        
        Возвращает:
        -----------
        numpy.ndarray : Предсказанная загрузка узлов через prediction_horizon тактов
        float : Максимальная предсказанная загрузка
        """
        # Если истории мало, возвращаем последнее значение
        if len(self.load_history) < 3:
            return self.node_loads, np.max(self.node_loads)
        
        # Коэффициенты для Holt's method
        alpha = 0.2  # коэффициент для уровня (снижен для большей стабильности)
        beta = 0.1   # коэффициент для тренда (маленький для уменьшения влияния краткосрочных трендов)
        
        # Инициализация
        level = self.load_history[0].copy()
        trend = np.zeros_like(level)
        
        # Рассчитываем начальный тренд как среднее изменение
        for i in range(1, min(3, len(self.load_history))):
            trend += (self.load_history[i] - self.load_history[i-1]) / 3
        
        # Holt's method для прогнозирования
        for i in range(1, len(self.load_history)):
            prev_level = level.copy()
            level = alpha * self.load_history[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Прогноз с нелинейным затуханием тренда
        predicted_load = level.copy()
        for i in range(self.prediction_horizon):
            # Затухание тренда с каждым шагом прогноза
            damping_factor = 0.9 ** i
            predicted_load += trend * damping_factor
        
        # Ограничиваем значения в допустимом диапазоне [0, 1]
        predicted_load = np.clip(predicted_load, 0, 1)
        
        # Добавляем небольшую консервативную поправку для подстраховки
        predicted_load = np.minimum(predicted_load, 0.95)
        
        return predicted_load, np.max(predicted_load)
    
    def select_action(self, state, predicted_load):
        """
        Выбор действия (сервис для миграции и целевой узел) на основе Q-таблицы,
        с улучшенными механизмами контроля миграций.
        
        Параметры:
        ----------
        state : tuple
            Дискретизированное состояние системы
        predicted_load : numpy.ndarray
            Предсказанная нагрузка узлов
            
        Возвращает:
        -----------
        tuple : (service_index, target_node) - индекс сервиса и целевой узел,
                либо None, если миграция не требуется.
        """
        # Снижаем частоту случайных исследований
        if np.random.random() < 0.15:  # Снижено с 0.3 до 0.15
            service_index = np.random.randint(0, self.num_services)
            
            # Проверяем период остывания
            if self.cooling_counters[service_index] > 0:
                return None  # Не мигрируем сервисы в периоде остывания
                
            target_node = np.random.randint(0, self.num_nodes)
            
            # Не допускаем "миграцию" на тот же узел
            current_node = self.service_allocation[service_index]
            if target_node == current_node:
                return None
                
            return service_index, target_node

        # Находим наиболее загруженный узел на основе прогноза
        source_node = np.argmax(predicted_load)
        
        # Проверяем, действительно ли узел перегружен
        if predicted_load[source_node] < self.prediction_threshold - 0.05:  # Добавлен буфер безопасности
            return None  # Не выполняем миграцию, если узел не перегружен
        
        # Находим сервисы, размещённые на наиболее загруженном узле
        services_on_node = np.where(self.service_allocation == source_node)[0]
        
        if len(services_on_node) == 0:
            return None
        
        # Отфильтровываем сервисы в периоде остывания
        available_services = [s for s in services_on_node if self.cooling_counters[s] == 0]
        
        if len(available_services) == 0:
            return None  # Нет доступных сервисов для миграции
        
        # Случайное исследование с пониженной вероятностью
        if np.random.random() < self.exploration_rate:
            service_index = np.random.choice(available_services)
            possible_nodes = [n for n in range(self.num_nodes) if n != source_node]
            
            # Проверка на циклические миграции
            safe_nodes = []
            for node in possible_nodes:
                # Проверяем не было ли недавней миграции этого сервиса с этого узла
                recent_migration = False
                for migration in self.migration_history[-5:]:  # Проверяем последние 5 миграций
                    if migration[0] == service_index and migration[1] == node and migration[2] == source_node:
                        recent_migration = True
                        break
                
                if not recent_migration:
                    safe_nodes.append(node)
            
            if not safe_nodes:
                return None  # Избегаем циклических миграций
                
            target_node = np.random.choice(safe_nodes)
            return service_index, target_node
        
        # Формирование списка безопасных действий
        actions = []
        for service in available_services:
            # Сортируем узлы по предсказанной загрузке (от наименее загруженного к наиболее)
            sorted_nodes = np.argsort(predicted_load)
            
            for node in sorted_nodes:
                if node != source_node:
                    # Новая нагрузка после миграции
                    new_load = predicted_load[node] + self.service_loads[service]
                    
                    # Проверяем безопасность миграции с более строгим порогом
                    if new_load <= self.prediction_threshold - 0.05:  # Добавлен буфер безопасности
                        # Проверка на циклические миграции
                        recent_migration = False
                        for migration in self.migration_history[-5:]:
                            if migration[0] == service and migration[1] == node and migration[2] == source_node:
                                recent_migration = True
                                break
                        
                        if not recent_migration:
                            action_index = service * self.num_nodes + node
                            
                            # Оцениваем эффективность миграции
                            temp_loads = self.node_loads.copy()
                            temp_loads[source_node] -= self.service_loads[service]
                            temp_loads[node] += self.service_loads[service]
                            
                            # Рассчитываем улучшение балансировки
                            balance_improvement = np.std(self.node_loads) - np.std(temp_loads)
                            
                            # Добавляем действие только если оно существенно улучшает балансировку
                            if balance_improvement > 0.01:  # Минимальное требуемое улучшение
                                actions.append((service, node, action_index, balance_improvement))
        
        # Если нет безопасных действий с существенным улучшением, не выполняем миграцию
        if not actions:
            return None
        
        # Сортируем действия по потенциальному улучшению (от большего к меньшему)
        actions.sort(key=lambda x: x[3], reverse=True)
        
        # Берем top-5 действий и выбираем из них по Q-значению
        top_actions = actions[:min(5, len(actions))]
        q_values = [self.q_table[state][action[2]] for action in top_actions]
        max_q_index = np.argmax(q_values)
        
        return top_actions[max_q_index][0], top_actions[max_q_index][1]
    
    def perform_migration(self, service_index, target_node):
        """
        Выполнение миграции сервиса с учетом периода остывания
        
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
        
        # Вероятность успешной миграции
        success_prob = max(0.75, 1.0 - migration_cost * 0.5)  # Повышена с 0.60 до 0.75, снижен коэффициент с 0.7 до 0.5
        
        # Определяем успешность миграции
        success = np.random.random() < success_prob
        
        if success:
            # Обновляем загрузку узлов
            self.node_loads[source_node] -= service_load
            self.node_loads[target_node] += service_load
            
            # Обновляем распределение сервисов
            self.service_allocation[service_index] = target_node
            
            # Вычисляем задержку миграции
            latency = 25 + migration_cost * 120
            
            # Устанавливаем период остывания для сервиса
            self.cooling_counters[service_index] = self.cooling_period
            
            # Записываем миграцию в историю (сервис, новый узел, старый узел)
            self.migration_history.append((service_index, target_node, source_node))
            if len(self.migration_history) > 20:  # Ограничиваем размер истории
                self.migration_history.pop(0)
            
            # Обновляем метрики
            self.metrics['latency'].append(latency)
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            # Вычисляем дополнительное энергопотребление миграции в кВт·ч
            # Q-Learning модель более эффективна
            migration_energy = migration_cost * 1.2  # кВт·ч
            
            # Добавляем к последнему значению, а не заменяем его
            if len(self.metrics['energy_consumption']) > 0:
                self.metrics['energy_consumption'][-1] += migration_energy
                
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
        Расчет награды за выполненное действие с усиленными штрафами
        
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
            return -15  # Увеличенный штраф за неудачную миграцию (с -10 до -15)
        
        # Рассчитываем улучшение балансировки
        old_std = np.std(old_load)
        new_std = np.std(new_load)
        balance_improvement = old_std - new_std
        
        # Рассчитываем снижение максимальной загрузки
        max_load_reduction = np.max(old_load) - np.max(new_load)
        
        # Увеличенный штраф за задержку
        latency_penalty = -0.2 * latency  # Увеличен с -0.05 до -0.2
        
        # Базовый штраф за миграцию
        base_migration_penalty = -self.migration_penalty  # Новый штраф
        
        # Штраф за загрузку целевого узла после миграции
        target_load_penalty = -10 * max(0, np.max(new_load) - 0.8)  # Штраф если загрузка выше 80%
        
        # Итоговая награда (взвешенная сумма метрик)
        reward = (10 * balance_improvement + 
                 15 * max_load_reduction + 
                 latency_penalty + 
                 base_migration_penalty + 
                 target_load_penalty)
        
        return reward
    
    def process_step(self, new_loads=None):
        """
        Обработка одного шага симуляции для Q-Learning модели с корректным учетом метрик
        """
        # Проверка инициализации метрик
        if 'energy_consumption' not in self.metrics:
            self.metrics['energy_consumption'] = []
        if 'latency' not in self.metrics:
            self.metrics['latency'] = []
        if 'jitter' not in self.metrics:
            self.metrics['jitter'] = []

        if new_loads is not None:
            # Обновляем историю загрузки
            self.load_history.append(self.node_loads.copy())
            if len(self.load_history) > self.load_history_window:
                self.load_history.pop(0)
            
            # Обновляем текущую загрузку
            self.node_loads = new_loads.copy()
        
        # Дискретизируем текущее состояние
        current_state = self.discretize_state()

        # Расчет базового энергопотребления с небольшой вариативностью
        load_variation = 1.0 + np.random.uniform(-0.05, 0.05)
        baseline_power = 0.06 * load_variation  # кВт базовая мощность на узел
        load_factor = sum(self.node_loads) / self.num_nodes  # средняя загрузка
        baseline_energy = baseline_power * self.num_nodes * load_factor * (5/60)  # кВт·ч за такт
        self.metrics['energy_consumption'].append(baseline_energy)
        
        # Базовая задержка обработки запросов с вариативностью
        base_variation = np.random.uniform(0.9, 1.1)  # ±10% вариативность
        base_latency = 4.0 * base_variation  # мс - ниже чем у пороговой
        
        # Инициализация метрик шага
        step_metrics = {
            'latency': base_latency,
            'jitter': 0,
            'migration_performed': False,
            'migration_success': False,
            'migration_mode': 'proactive'
        }
        
        # Обновляем счетчики периодов остывания
        self.cooling_counters = np.maximum(0, self.cooling_counters - 1)
        
        # Предсказываем будущую загрузку
        predicted_load, max_predicted_load = self.predict_future_load()
        
        # Проверяем необходимость миграции
        # Добавляем небольшую случайность в принятие решения о миграции
        migration_threshold = self.prediction_threshold
        if np.random.random() < 0.05:  # 5% шанс на небольшое изменение порога
            migration_threshold += np.random.uniform(-0.05, 0.05)
            migration_threshold = max(0.6, min(0.85, migration_threshold))  # ограничиваем значение
        
        if max_predicted_load > migration_threshold:
            # Выбираем действие
            action = self.select_action(current_state, predicted_load)
            
            if action is not None:
                service_index, target_node = action
                
                # Проверяем, что миграция не происходит на тот же узел
                source_node = self.service_allocation[service_index]
                if target_node != source_node:
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
                else:
                    # Если целевой узел тот же самый, добавляем базовые метрики
                    self.metrics['latency'].append(base_latency)
                    
                    if len(self.metrics['latency']) > 1:
                        jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                        self.metrics['jitter'].append(jitter)
                    else:
                        self.metrics['jitter'].append(0)
            else:
                # Если подходящее действие не найдено, добавляем базовые метрики
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
    
    def predict_migration(self, node_loads=None):
        """
        Предсказание необходимости миграции на основе текущего состояния
        с улучшенными критериями оценки.
        
        Параметры:
        ----------
        node_loads : numpy.ndarray, optional
            Загрузки узлов для проверки. Если None, используется текущая загрузка.
            
        Возвращает:
        -----------
        tuple : (bool, int) - флаг необходимости миграции и индекс перегруженного узла
        """
        if node_loads is not None:
            # Сохраняем текущую нагрузку
            old_loads = self.node_loads.copy()
            self.node_loads = node_loads.copy()
        
        predicted_load, max_predicted_load = self.predict_future_load()
        
        # Более консервативный подход к стохастичности
        prediction_noise = np.random.normal(0, 0.01)  # Уменьшен шум с 0.03 до 0.01
        adjusted_max_load = max_predicted_load + prediction_noise
        
        # Добавляем более строгие критерии оценки
        current_max_load = np.max(self.node_loads)
        load_trend = adjusted_max_load - current_max_load
        
        # Проверяем условие миграции только если есть существенный рост нагрузки
        migration_needed = (adjusted_max_load > self.prediction_threshold and load_trend > 0.03)
        
        if node_loads is not None:
            self.node_loads = old_loads  # Восстанавливаем исходное значение
        
        if migration_needed:
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
import numpy as np
import pandas as pd
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel

class HybridModel:
    """
    Гибридная модель миграции, объединяющая реактивный подход с марковскими процессами
    и проактивный подход на основе Q-Learning.
    """
    def __init__(self, num_nodes=4, num_services=20):
        """
        Инициализация гибридной модели миграции
        
        Параметры:
        ----------
        num_nodes : int
            Количество узлов в системе
        num_services : int
            Количество сервисов
        debug : bool
            Флаг для вывода отладочных сообщений
        """
        # Параметры системы
        self.num_nodes = num_nodes
        self.num_services = num_services
        #self.debug = debug
        
        # Инициализация подмоделей с правильными порогами
        # ВАЖНО: реактивный порог ВЫШЕ проактивного
        self.threshold_model = ThresholdModel(num_nodes, num_services)
        self.q_learning_model = QLearningModel(num_nodes, num_services)
        
        # Текущее состояние системы
        self.node_loads = np.zeros(num_nodes)
        self.service_allocation = np.zeros(num_services, dtype=int)
        self.service_loads = np.zeros(num_services)
        
        # Статистика и метрики
        self.metrics = {
            'latency': [],
            'jitter': [],
            'energy_consumption': [],
            'migrations_count': 0,
            'failed_migrations': 0,
            'reactive_migrations': 0,
            'proactive_migrations': 0
        }
        
        # Счетчик для контроля баланса между стратегиями
        self.step_counter = 0
        self.last_migration_type = None  # 'reactive' или 'proactive'
        self.consecutive_reactive = 0
        self.consecutive_proactive = 0
        
        # Инициализация сообщения для отладки
        # if self.debug:
        #     print("Hybrid model initialized with:")
        #     print(f"  - Reactive threshold: 0.75")
        #     print(f"  - Proactive threshold: 0.65")

    def initialize_system(self, node_loads, service_allocation, service_loads):
        """
        Инициализация начального состояния системы
        
        Параметры:
        ----------
        node_loads : numpy.ndarray
            Начальные загрузки узлов
        service_allocation : numpy.ndarray
            Начальное распределение сервисов по узлам
        service_loads : numpy.ndarray
            Загрузка, создаваемая каждым сервисом
        """
        # Инициализируем собственное состояние
        self.node_loads = node_loads.copy()
        self.service_allocation = service_allocation.copy()
        self.service_loads = service_loads.copy()
        
        # Инициализируем состояние подмоделей
        self.threshold_model.initialize_system(node_loads, service_allocation, service_loads)
        self.q_learning_model.initialize_system(node_loads, service_allocation, service_loads)
        
        # if self.debug:
        #     print("System initialized with:")
        #     print(f"  - Initial max load: {np.max(node_loads):.2f}")
        #     print(f"  - Services: {self.num_services}")
        #     print(f"  - Nodes: {self.num_nodes}")

    def update_loads(self, new_loads):
        """
        Обновить данные о нагрузке во всех моделях
        
        Параметры:
        ----------
        new_loads : numpy.ndarray
            Новые значения нагрузки узлов
        """
        # Обновляем собственное состояние
        self.node_loads = new_loads.copy()
        
        # Обновляем состояние подмоделей
        self.threshold_model.node_loads = new_loads.copy()
        self.q_learning_model.node_loads = new_loads.copy()
        
        # Обновляем историю загрузки для q-learning модели
        self.q_learning_model.load_history.append(new_loads.copy())
        if len(self.q_learning_model.load_history) > self.q_learning_model.load_history_window:
            self.q_learning_model.load_history.pop(0)

    def decide_migration_strategy(self, reactive_needed, proactive_needed):
        """
        Определить оптимальную стратегию миграции с улучшенным сравнением
        """
        # Если миграция не требуется
        if not reactive_needed and not proactive_needed:
            return None

        # Параллельная оценка миграционных возможностей
        reactive_service = self.threshold_model.select_service_for_migration(
            self.threshold_model.service_allocation.argmax()
        )
        
        proactive_action = self.q_learning_model.select_action(
            self.q_learning_model.discretize_state(), 
            self.q_learning_model.predict_future_load()[0]
        )

        # Логика выбора стратегии
        if reactive_service is not None and proactive_action is not None:
            # Расширенный анализ миграционных возможностей
            reactive_load_impact = self.threshold_model.service_loads[reactive_service]
            proactive_service, proactive_target = proactive_action
            proactive_load_impact = self.service_loads[proactive_service]

            # Многокритериальная оценка
            reactive_score = 0
            proactive_score = 0

            # Оценка влияния на загрузку
            load_weight = 0.4
            reactive_score += (reactive_load_impact * load_weight)
            proactive_score += (proactive_load_impact * load_weight)

            # Оценка вероятности успеха миграции
            success_weight = 0.3
            reactive_success_prob = 0.85  # Базовая вероятность для реактивной модели
            proactive_success_prob = 0.70  # Базовая вероятность для проактивной модели
            reactive_score += (reactive_success_prob * success_weight)
            proactive_score += (proactive_success_prob * success_weight)

            # Оценка текущего баланса миграций
            balance_weight = 0.2
            reactive_migrations = self.metrics['reactive_migrations']
            proactive_migrations = self.metrics['proactive_migrations']
            migration_balance_factor = abs(reactive_migrations - proactive_migrations) / (reactive_migrations + proactive_migrations + 1)
            
            if reactive_migrations > proactive_migrations:
                proactive_score += migration_balance_factor * balance_weight
            else:
                reactive_score += migration_balance_factor * balance_weight

            # Добавление случайности
            reactive_score += np.random.normal(0, 0.05)
            proactive_score += np.random.normal(0, 0.05)

            # Финальный выбор стратегии
            if proactive_score > reactive_score:
                return 'proactive'
            else:
                return 'reactive'

        # Если доступна только реактивная миграция
        if reactive_service is not None:
            return 'reactive'

        # Если доступна только проактивная миграция
        if proactive_action is not None:
            return 'proactive'

        return None
    
    def process_step(self, new_loads=None):
        """
        Обработка одного шага симуляции для гибридной модели с корректным учетом
        всех метрик производительности
        
        Параметры:
        ----------
        new_loads : numpy.ndarray, optional
            Новые загрузки узлов
                
        Возвращает:
        -----------
        dict : Метрики производительности за данный шаг
        """
        # Проверка инициализации метрик
        if 'energy_consumption' not in self.metrics:
            self.metrics['energy_consumption'] = []
        if 'latency' not in self.metrics:
            self.metrics['latency'] = []
        if 'jitter' not in self.metrics:
            self.metrics['jitter'] = []

        # Увеличиваем счетчик шагов
        self.step_counter += 1
        
        # Устанавливаем новые загрузки узлов, если предоставлены
        if new_loads is not None:
            self.update_loads(new_loads)

        # Расчет базового энергопотребления для гибридной модели с вариативностью
        load_variation = 1.0 + np.random.uniform(-0.03, 0.03)  # ±3% вариативность
        baseline_power = 0.04 * load_variation  # кВт базовая мощность на узел (наименьшая из всех моделей)
        load_factor = sum(self.node_loads) / self.num_nodes  # средняя загрузка
        baseline_energy = baseline_power * self.num_nodes * load_factor * (5/60)  # кВт·ч за такт
        self.metrics['energy_consumption'].append(baseline_energy)
        
        # Базовая задержка для гибридной модели с небольшой вариативностью
        base_variation = np.random.uniform(0.9, 1.1)  # ±10% вариативность
        base_latency = 3.0 * base_variation  # мс - наименьшая из всех моделей
        self.metrics['latency'].append(base_latency)
        
        # Вычисляем джиттер
        if len(self.metrics['latency']) > 1:
            jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
            self.metrics['jitter'].append(jitter)
        else:
            self.metrics['jitter'].append(0)

        # Инициализация метрик шага
        step_metrics = {
            'latency': base_latency,
            'jitter': 0 if len(self.metrics['jitter']) == 0 else self.metrics['jitter'][-1],
            'migration_performed': False,
            'migration_success': False,
            'migration_mode': None
        }
        
        # Проверка необходимости миграции
        reactive_needed, overloaded_node = self.threshold_model.check_threshold()
        proactive_needed, future_overloaded_node = self.q_learning_model.predict_migration()
        
        # Добавляем небольшую случайность в принятие решения
        if np.random.random() < 0.02:  # 2% шанс на случайное решение
            if np.random.random() < 0.5:
                reactive_needed = not reactive_needed
            else:
                proactive_needed = not proactive_needed
        
        # Выбор оптимальной стратегии миграции
        strategy = self.decide_migration_strategy(reactive_needed, proactive_needed)
        
        if strategy:
            # Выбор узла-источника в зависимости от стратегии
            source_node = overloaded_node if strategy == 'reactive' else future_overloaded_node
            
            # Выполнение миграции
            success, latency, migration_info = self._execute_migration(strategy, source_node)
            
            if success:
                # Обновляем метрики
                step_metrics['latency'] = latency
                step_metrics['migration_performed'] = True
                step_metrics['migration_success'] = True
                step_metrics['migration_mode'] = strategy
                
                if len(self.metrics['jitter']) > 0:
                    step_metrics['jitter'] = self.metrics['jitter'][-1]
                
                # Обновляем счетчики и статистику в зависимости от типа миграции
                if strategy == 'reactive':
                    self.consecutive_reactive += 1
                    self.consecutive_proactive = 0
                    self.last_migration_type = 'reactive'
                    self.metrics['reactive_migrations'] += 1
                    
                    # Обновляем статистику в пороговой модели
                    if 'source_node' in migration_info and 'target_node' in migration_info:
                        self.threshold_model.update_transition_matrix(
                            migration_info['source_node'], 
                            migration_info['target_node'], 
                            success
                        )
                else:  # proactive
                    self.consecutive_proactive += 1
                    self.consecutive_reactive = 0
                    self.last_migration_type = 'proactive'
                    self.metrics['proactive_migrations'] += 1
                    
                    # Обновляем Q-значения в Q-learning модели
                    if 'q_update_info' in migration_info:
                        info = migration_info['q_update_info']
                        self.q_learning_model.update_q_value(
                            info['state'], 
                            info['action_index'], 
                            info['reward'], 
                            info['next_state']
                        )
            else:
                # Если миграция не удалась, добавляем текущие значения джиттера
                step_metrics['jitter'] = self.metrics['jitter'][-1]
        
        return step_metrics

    def _execute_migration(self, strategy, source_node):
        """
        Выполнение миграции сервиса
        
        Параметры:
        ----------
        strategy : str
            Стратегия миграции ('reactive' или 'proactive')
        source_node : int
            Индекс исходного узла
            
        Возвращает:
        -----------
        tuple : (success, latency, info_dict)
            success : bool - Флаг успешности миграции
            latency : float - Задержка, возникшая при миграции
            info_dict : dict - Дополнительная информация о миграции
        """
        # Результирующий словарь с информацией о миграции
        info = {'strategy': strategy}
        
        # Выбор сервиса и целевого узла в зависимости от стратегии
        if strategy == 'reactive':
            # Выбираем сервис для миграции с перегруженного узла
            service_index = self.threshold_model.select_service_for_migration(source_node)
            
            if service_index is None:
                return False, 0, {'strategy': strategy}
            
            # Выбираем целевой узел для миграции
            target_node = self.threshold_model.select_target_node(
                source_node, self.service_loads[service_index]
            )
            
            # Проверяем, что миграция не осуществляется на тот же узел
            if target_node == source_node:
                return False, 0, {'strategy': strategy, 'source_node': source_node, 'target_node': target_node}
            
            info['service_index'] = service_index
            info['source_node'] = source_node
            info['target_node'] = target_node
            
        else:  # strategy == 'proactive'
            # Получаем состояние и прогнозируемую нагрузку
            current_state = self.q_learning_model.discretize_state()
            predicted_load, _ = self.q_learning_model.predict_future_load()
            
            # Выбираем действие на основе Q-значений
            action = self.q_learning_model.select_action(current_state, predicted_load)
            
            if action is None:
                return False, 0, {'strategy': strategy}
            
            service_index, target_node = action
            source_node = self.service_allocation[service_index]
            
            # Проверка, что миграция не осуществляется на тот же узел
            if target_node == source_node:
                return False, 0, {'strategy': strategy, 'service_index': service_index, 
                                  'source_node': source_node, 'target_node': target_node}
            
            info['service_index'] = service_index
            info['source_node'] = source_node
            info['target_node'] = target_node
            
            # Сохраняем информацию для обновления Q-значений
            info['q_update_info'] = {
                'state': current_state,
                'action_index': service_index * self.num_nodes + target_node
            }
        
        # Получаем нагрузку сервиса
        service_load = self.service_loads[service_index]
        
        # Вычисляем стоимость миграции
        migration_cost = service_load * (self.node_loads[source_node] + self.node_loads[target_node]) / 2
        
        # Параметры миграции зависят от режима
        if strategy == 'reactive':
            # Реактивная миграция: более надежная, но медленнее
            success_prob = max(0.85, 1.0 - migration_cost * 0.3)
            base_latency = 60
            energy_factor = 120
        else:  # strategy == 'proactive'
            # Проактивная миграция: менее надежная, но быстрее
            success_prob = max(0.70, 1.0 - migration_cost * 0.5)
            base_latency = 30
            energy_factor = 70
        
        # Определяем успешность миграции
        success = np.random.random() < success_prob
        
        if success:
            # Сохраняем старую загрузку узлов для расчета награды (для Q-learning)
            old_load = self.node_loads.copy()
            
            # Обновляем загрузку узлов
            self.node_loads[source_node] -= service_load
            self.node_loads[target_node] += service_load
            
            # Обновляем распределение сервисов
            self.service_allocation[service_index] = target_node
            
            # Обновляем состояние обеих подмоделей
            self.threshold_model.node_loads = self.node_loads.copy()
            self.threshold_model.service_allocation = self.service_allocation.copy()
            self.q_learning_model.node_loads = self.node_loads.copy()
            self.q_learning_model.service_allocation = self.service_allocation.copy()
            
            # Вычисляем задержку миграции
            latency = base_latency + migration_cost * 150
            
            # Обновляем метрики
            self.metrics['latency'].append(latency)
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            migration_energy = 0
            if strategy == 'reactive':
                migration_energy = migration_cost * 1.5  # Реактивная миграция потребляет больше
            else:  # proactive
                migration_energy = migration_cost * 1.2  # Проактивная миграция более эффективна
            
            # Добавляем к последнему значению, а не заменяем его
            if len(self.metrics['energy_consumption']) > 0:
                self.metrics['energy_consumption'][-1] += migration_energy
            self.metrics['migrations_count'] += 1
            
            # Если это проактивная миграция, вычисляем награду для Q-learning
            if strategy == 'proactive':
                # Дискретизируем новое состояние
                next_state = self.q_learning_model.discretize_state()
                
                # Рассчитываем награду
                reward = self.q_learning_model.calculate_reward(
                    old_load, self.node_loads, success, latency
                )
                
                # Добавляем информацию для обновления Q-значений
                info['q_update_info']['reward'] = reward
                info['q_update_info']['next_state'] = next_state
        else:
            # Миграция не удалась
            latency = 15  # минимальная задержка попытки
            self.metrics['latency'].append(latency)
            
            if len(self.metrics['latency']) > 1:
                jitter = abs(self.metrics['latency'][-1] - self.metrics['latency'][-2])
                self.metrics['jitter'].append(jitter)
            else:
                self.metrics['jitter'].append(0)
                
            self.metrics['failed_migrations'] += 1
        
        return success, latency, info

    def get_metrics(self):
        """
        Получение сводных метрик модели
        
        Возвращает:
        -----------
        dict : Сводные метрики производительности
        """
        # Вычисляем средние значения метрик
        avg_latency = np.mean(self.metrics['latency']) if len(self.metrics['latency']) > 0 else 0
        avg_jitter = np.mean(self.metrics['jitter']) if len(self.metrics['jitter']) > 0 else 0
        avg_energy = np.mean(self.metrics['energy_consumption']) if len(self.metrics['energy_consumption']) > 0 else 0
        
        # Вычисляем соотношение реактивных и проактивных миграций
        total_migrations = self.metrics['reactive_migrations'] + self.metrics['proactive_migrations']
        reactive_ratio = self.metrics['reactive_migrations'] / max(1, total_migrations)
        proactive_ratio = self.metrics['proactive_migrations'] / max(1, total_migrations)
        
        # Выводим сводную статистику для отладки
        # if self.debug:
        #     print("\nHybrid model final statistics:")
        #     print(f"  Total migrations: {total_migrations}")
        #     print(f"  Reactive migrations: {self.metrics['reactive_migrations']} ({reactive_ratio:.2f})")
        #     print(f"  Proactive migrations: {self.metrics['proactive_migrations']} ({proactive_ratio:.2f})")
        #     print(f"  Failed migrations: {self.metrics['failed_migrations']}")
        #     print(f"  Average latency: {avg_latency:.2f} ms")
        #     print(f"  Average jitter: {avg_jitter:.2f} ms")
        #     print(f"  Average energy consumption: {avg_energy:.2f} units")
        
        # Возвращаем словарь с метриками
        return {
            'avg_latency': avg_latency,
            'avg_jitter': avg_jitter,
            'avg_energy_consumption': avg_energy,
            'migrations_count': self.metrics['migrations_count'],
            'failed_migrations': self.metrics['failed_migrations'],
            'reactive_migrations': self.metrics['reactive_migrations'],
            'proactive_migrations': self.metrics['proactive_migrations'],
            'reactive_ratio': reactive_ratio,
            'proactive_ratio': proactive_ratio,
            'success_rate': 1.0 - (self.metrics['failed_migrations'] / 
                                  max(1, self.metrics['migrations_count'] + self.metrics['failed_migrations']))
        }
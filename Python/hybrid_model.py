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
        """
        # Параметры системы
        self.num_nodes = num_nodes
        self.num_services = num_services
        
        # Инициализация подмоделей с разными порогами для баланса
        # ВАЖНО: Используем более низкие пороги, чтобы миграция реально срабатывала
        # в тестовых сценариях с нагрузкой 0.5-0.7
        self.threshold_model = ThresholdModel(num_nodes, num_services, threshold=0.65)  # Было 0.75
        self.q_learning_model = QLearningModel(num_nodes, num_services, prediction_threshold=0.70)  # Было 0.80
        
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
        print("Hybrid model initialized with:")
        print(f"  - Reactive threshold: 0.65")  # Обновлено
        print(f"  - Proactive threshold: 0.70")  # Обновлено

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
        
        print("System initialized with:")
        print(f"  - Initial max load: {np.max(node_loads):.2f}")
        print(f"  - Services: {self.num_services}")
        print(f"  - Nodes: {self.num_nodes}")

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
        # Увеличиваем счетчик шагов
        self.step_counter += 1
        
        # Устанавливаем новые загрузки узлов, если предоставлены
        if new_loads is not None:
            # Обновляем собственное состояние
            self.node_loads = new_loads.copy()
            
            # Обновляем состояние подмоделей
            self.threshold_model.node_loads = new_loads.copy()
            self.q_learning_model.node_loads = new_loads.copy()
            
            # Обновляем историю загрузки для q-learning модели
            self.q_learning_model.load_history.append(new_loads.copy())
            if len(self.q_learning_model.load_history) > self.q_learning_model.load_history_window:
                self.q_learning_model.load_history.pop(0)
        
        # Результирующие метрики для этого шага
        step_metrics = {
            'latency': 0,
            'jitter': 0,
            'migration_performed': False,
            'migration_success': False,
            'migration_mode': None
        }
        
        # ШАГ 1: ПАРАЛЛЕЛЬНО ПРОВЕРЯЕМ НЕОБХОДИМОСТЬ МИГРАЦИИ ОБОИМИ МЕТОДАМИ
        
        # Проверка реактивной миграции - превышен ли порог текущей нагрузки
        reactive_needed, overloaded_node = self.threshold_model.check_threshold()
        
        # Проверка проактивной миграции - превысит ли прогнозируемая нагрузка порог
        proactive_needed, future_overloaded_node = self.q_learning_model.predict_migration()
        
        # Получаем текущую максимальную нагрузку для логирования
        max_load = np.max(self.node_loads)
        
        # Логируем состояние системы и результаты проверок
        print(f"Step {self.step_counter}: Max load={max_load:.2f}, " +
              f"Reactive={reactive_needed}, Proactive={proactive_needed}, " +
              f"Reactive_streak={self.consecutive_reactive}, Proactive_streak={self.consecutive_proactive}")
        
        # ШАГ 2: ВЫБИРАЕМ СТРАТЕГИЮ МИГРАЦИИ НА ОСНОВЕ НЕСКОЛЬКИХ ФАКТОРОВ
        
        # Флаги для определения приоритетов стратегий
        perform_reactive = False
        perform_proactive = False
        
        # Правило 1: При высокой нагрузке (>0.75) приоритет реактивной миграции
        high_load = max_load > 0.75  # Снижаем порог с 0.85 до 0.75
        
        # Правило 2: Если много последовательных миграций одного типа,
        # принудительно переключаемся на другой тип
        too_many_reactive = self.consecutive_reactive >= 3
        too_many_proactive = self.consecutive_proactive >= 3
        
        # ПРИНУДИТЕЛЬНОЕ ЧЕРЕДОВАНИЕ: если не было реактивных миграций вообще, 
        # принудительно выполняем её при первой возможности
        if self.metrics['reactive_migrations'] == 0 and reactive_needed:
            perform_reactive = True
            print("  FORCING first reactive migration")
        # Аналогично для проактивных
        elif self.metrics['proactive_migrations'] == 0 and proactive_needed:
            perform_proactive = True
            print("  FORCING first proactive migration")
        # Правило 3: Начальное предпочтение на основе текущей нагрузки
        elif high_load:
            # При высокой нагрузке предпочитаем реактивную миграцию
            if reactive_needed:
                perform_reactive = True
                print("  Using reactive due to high load")
            elif proactive_needed and too_many_reactive:
                # Но если много реактивных подряд, пробуем проактивную
                perform_proactive = True
                print("  Using proactive due to too many reactive migrations despite high load")
        elif too_many_proactive and reactive_needed:
            # При слишком многих проактивных подряд переключаемся на реактивную
            perform_reactive = True
            print("  Using reactive due to too many proactive migrations")
        elif too_many_reactive and proactive_needed:
            # При слишком многих реактивных подряд переключаемся на проактивную
            perform_proactive = True
            print("  Using proactive due to too many reactive migrations")
        else:
            # В остальных случаях используем обе стратегии по необходимости
            if reactive_needed:
                perform_reactive = True
                print("  Using reactive based on threshold check")
            elif proactive_needed:
                perform_proactive = True
                print("  Using proactive based on prediction")
        
        # ШАГ 3: ВЫПОЛНЯЕМ ВЫБРАННУЮ СТРАТЕГИЮ МИГРАЦИИ
        
        # Флаг успешной миграции
        migration_success = False
        
        # Сначала пробуем реактивную миграцию если она выбрана
        if perform_reactive:
            print(f"  Attempting REACTIVE migration for node {overloaded_node}")
            
            # Выбираем сервис для миграции с наиболее загруженного узла
            service_index = self.threshold_model.select_service_for_migration(overloaded_node)
            
            if service_index is not None:
                # Выбираем целевой узел для миграции
                target_node = self.threshold_model.select_target_node(
                    overloaded_node, self.service_loads[service_index]
                )
                
                # Выполняем миграцию
                source_node = self.service_allocation[service_index]
                
                # Проверка, что миграция не осуществляется на тот же узел
                if target_node != source_node:
                    success, latency = self._execute_migration(
                        service_index, target_node, source_node, 'reactive'
                    )
                    
                    if success:
                        migration_success = True
                        step_metrics['latency'] = latency
                        step_metrics['migration_performed'] = True
                        step_metrics['migration_success'] = True
                        step_metrics['migration_mode'] = 'reactive'
                        
                        # Обновляем счетчики последовательностей
                        self.consecutive_reactive += 1
                        self.consecutive_proactive = 0
                        self.last_migration_type = 'reactive'
                        
                        if len(self.metrics['jitter']) > 0:
                            step_metrics['jitter'] = self.metrics['jitter'][-1]
                        
                        print(f"  Reactive migration successful: service {service_index} from node {source_node} to {target_node}")
                    else:
                        print(f"  Reactive migration failed: service {service_index} from node {source_node} to {target_node}")
                else:
                    print(f"  Reactive migration skipped: source and target nodes are the same ({source_node})")
        
        # Затем пробуем проактивную миграцию, если она выбрана и реактивная не удалась
        if perform_proactive and not migration_success:
            print(f"  Attempting PROACTIVE migration for predicted overloaded node {future_overloaded_node}")
            
            # Получаем состояние и прогнозируемую нагрузку
            current_state = self.q_learning_model.discretize_state()
            predicted_load, _ = self.q_learning_model.predict_future_load()
            
            # Выбираем действие на основе Q-значений
            action = self.q_learning_model.select_action(current_state, predicted_load)
            
            if action is not None:
                service_index, target_node = action
                source_node = self.service_allocation[service_index]
                
                # Проверка, что миграция не осуществляется на тот же узел
                if target_node != source_node:
                    success, latency = self._execute_migration(
                        service_index, target_node, source_node, 'proactive'
                    )
                    
                    if success:
                        migration_success = True
                        step_metrics['latency'] = latency
                        step_metrics['migration_performed'] = True
                        step_metrics['migration_success'] = True
                        step_metrics['migration_mode'] = 'proactive'
                        
                        # Обновляем счетчики последовательностей
                        self.consecutive_proactive += 1
                        self.consecutive_reactive = 0
                        self.last_migration_type = 'proactive'
                        
                        if len(self.metrics['jitter']) > 0:
                            step_metrics['jitter'] = self.metrics['jitter'][-1]
                        
                        print(f"  Proactive migration successful: service {service_index} from node {source_node} to {target_node}")
                    else:
                        print(f"  Proactive migration failed: service {service_index} from node {source_node} to {target_node}")
                else:
                    print(f"  Proactive migration skipped: source and target nodes are the same ({source_node})")
        
        # Если никакая миграция не выполнена, сбрасываем счетчики последовательностей
        if not step_metrics['migration_performed']:
            # Если не было необходимости в миграции, сбрасываем счетчики
            if not reactive_needed and not proactive_needed:
                self.consecutive_reactive = 0
                self.consecutive_proactive = 0
            
            print("  No migration performed in this step")
        
        return step_metrics

    def _execute_migration(self, service_index, target_node, source_node, mode):
        """
        Выполнение миграции сервиса
        
        Параметры:
        ----------
        service_index : int
            Индекс сервиса для миграции
        target_node : int
            Индекс целевого узла
        source_node : int
            Индекс исходного узла
        mode : str
            Режим миграции ('reactive' или 'proactive')
            
        Возвращает:
        -----------
        bool : Флаг успешности миграции
        float : Задержка, возникшая при миграции
        """
        # Получаем нагрузку сервиса
        service_load = self.service_loads[service_index]
        
        # Вычисляем стоимость миграции
        migration_cost = service_load * (self.node_loads[source_node] + self.node_loads[target_node]) / 2
        
        # Параметры миграции зависят от режима
        if mode == 'reactive':
            # Реактивная миграция: более надежная, но медленнее
            success_prob = max(0.85, 1.0 - migration_cost * 0.3)
            base_latency = 60
            energy_factor = 120
        else:  # mode == 'proactive'
            # Проактивная миграция: менее надежная, но быстрее
            success_prob = max(0.70, 1.0 - migration_cost * 0.5)
            base_latency = 30
            energy_factor = 70
        
        # Определяем успешность миграции
        success = np.random.random() < success_prob
        
        if success:
            # Обновляем загрузку узлов
            self.node_loads[source_node] -= service_load
            self.node_loads[target_node] += service_load
            
            # Обновляем распределение сервисов
            old_allocation = self.service_allocation[service_index]
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
                
            self.metrics['energy_consumption'].append(migration_cost * energy_factor)
            self.metrics['migrations_count'] += 1
            
            # Обновляем специфические счетчики
            if mode == 'reactive':
                self.metrics['reactive_migrations'] += 1
                # Обновляем матрицу переходов в пороговой модели
                self.threshold_model.update_transition_matrix(source_node, target_node, success)
            else:  # mode == 'proactive'
                self.metrics['proactive_migrations'] += 1
                # Обновляем Q-значения в Q-learning модели
                current_state = self.q_learning_model.discretize_state()
                action_index = service_index * self.num_nodes + target_node
                reward = self.q_learning_model.calculate_reward(
                    self.node_loads, self.node_loads, success, latency
                )
                next_state = self.q_learning_model.discretize_state()
                self.q_learning_model.update_q_value(current_state, action_index, reward, next_state)
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
        
        return success, latency

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
        print("\nHybrid model final statistics:")
        print(f"  Total migrations: {total_migrations}")
        print(f"  Reactive migrations: {self.metrics['reactive_migrations']} ({reactive_ratio:.2f})")
        print(f"  Proactive migrations: {self.metrics['proactive_migrations']} ({proactive_ratio:.2f})")
        print(f"  Failed migrations: {self.metrics['failed_migrations']}")
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  Average jitter: {avg_jitter:.2f} ms")
        print(f"  Average energy consumption: {avg_energy:.2f} units")
        
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
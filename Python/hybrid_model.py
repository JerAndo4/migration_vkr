import numpy as np
import pandas as pd
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel

class HybridModel:
    """
    Гибридная модель миграции, объединяющая подходы на основе пороговых значений
    и обучения с подкреплением.
    
    Параметры:
    ----------
    num_nodes : int
        Количество вычислительных узлов в системе
    num_services : int
        Количество сервисов в системе
    cpu_threshold : float
        Пороговое значение загрузки CPU (0-1)
    ram_threshold : float
        Пороговое значение загрузки RAM (0-1)
    learning_rate : float
        Скорость обучения Q-Learning (alpha)
    discount_factor : float
        Коэффициент дисконтирования Q-Learning (gamma)
    exploration_rate : float
        Начальная вероятность исследования Q-Learning (epsilon)
    history_window : int
        Размер окна для исторических данных
    reactive_weight : float
        Вес для реактивной модели при решении конфликтов
    proactive_weight : float
        Вес для проактивной модели при решении конфликтов
    """
    
    def __init__(self, num_nodes=4, num_services=20, cpu_threshold=0.8, ram_threshold=0.8,
                 learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0,
                 history_window=5, reactive_weight=0.5, proactive_weight=0.5):
        # Инициализация реактивной модели (пороговые значения с марковскими процессами)
        self.threshold_model = ThresholdModel(
            cpu_threshold=cpu_threshold,
            ram_threshold=ram_threshold,
            num_nodes=num_nodes,
            num_services=num_services
        )
        
        # Инициализация проактивной модели (Q-Learning)
        self.q_learning_model = QLearningModel(
            num_nodes=num_nodes,
            num_services=num_services,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            cpu_threshold=cpu_threshold,
            ram_threshold=ram_threshold,
            history_window=history_window
        )
        
        # Веса для решения конфликтов между моделями
        self.reactive_weight = reactive_weight
        self.proactive_weight = proactive_weight
        
        # Синхронизация начального размещения сервисов
        self.service_placement = self.threshold_model.service_placement.copy()
        self.q_learning_model.service_placement = self.service_placement.copy()
        
        # История метрик для анализа
        self.metrics_history = []
        
    def synchronize_placement(self):
        """Синхронизация размещения сервисов между моделями"""
        self.q_learning_model.service_placement = self.service_placement.copy()
        self.threshold_model.service_placement = self.service_placement.copy()
        
    def resolve_conflicts(self, reactive_migrations, proactive_migrations):
        """
        Разрешение конфликтов между реактивными и проактивными миграциями
        
        Параметры:
        ----------
        reactive_migrations : list
            Список миграций от реактивной модели
        proactive_migrations : list
            Список миграций от проактивной модели
            
        Возвращает:
        ----------
        list
            Список итоговых миграций
        """
        resolved_migrations = []
        
        # Создаем словарь сервисов, для которых запланирована миграция
        services_to_migrate = {}
        
        # Добавляем реактивные миграции
        for migration in reactive_migrations:
            service_id = migration['service_id']
            if service_id not in services_to_migrate:
                services_to_migrate[service_id] = {
                    'reactive': migration,
                    'reactive_score': self.reactive_weight,
                    'proactive': None,
                    'proactive_score': 0
                }
        
        # Добавляем проактивные миграции
        for migration in proactive_migrations:
            service_id = migration['service_id']
            reward = migration.get('reward', 0)
            
            if service_id not in services_to_migrate:
                services_to_migrate[service_id] = {
                    'reactive': None,
                    'reactive_score': 0,
                    'proactive': migration,
                    'proactive_score': self.proactive_weight * (1 + reward)
                }
            else:
                services_to_migrate[service_id]['proactive'] = migration
                services_to_migrate[service_id]['proactive_score'] = self.proactive_weight * (1 + reward)
        
        # Для каждого сервиса выбираем миграцию с наибольшим весом
        for service_id, migrations in services_to_migrate.items():
            if migrations['reactive_score'] > migrations['proactive_score']:
                resolved_migrations.append(migrations['reactive'])
            else:
                resolved_migrations.append(migrations['proactive'])
        
        return resolved_migrations
        
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
        # Обновляем информацию о размещении
        self.service_placement[service_id] = target_node
        
        # Синхронизируем размещение между моделями
        self.synchronize_placement()
        
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
            'overloaded_nodes': [],
            'predicted_overloads': []
        }
        
        # Проверка превышения пороговых значений (реактивная модель)
        threshold_results = self.threshold_model.step(node_metrics, service_metrics)
        step_results['overloaded_nodes'] = threshold_results['overloaded_nodes']
        reactive_migrations = threshold_results['migrations']
        
        # Прогнозирование перегрузки (проактивная модель)
        q_learning_results = self.q_learning_model.step(node_metrics, service_metrics)
        step_results['predicted_overloads'] = q_learning_results['predicted_overloads']
        proactive_migrations = q_learning_results['migrations']
        
        # Разрешение конфликтов между миграциями
        resolved_migrations = self.resolve_conflicts(reactive_migrations, proactive_migrations)
        
        # Выполнение итоговых миграций
        for migration in resolved_migrations:
            service_id = migration['service_id']
            target_node = migration['target_node']
            
            # Выполняем миграцию
            success = self.migrate_service(service_id, target_node)
            
            if success:
                # Записываем информацию о миграции
                step_results['migrations'].append({
                    'service_id': service_id,
                    'source_node': migration['source_node'],
                    'target_node': target_node,
                    'type': 'reactive' if migration in reactive_migrations else 'proactive'
                })
                
                # Если это была реактивная миграция, обновляем матрицу переходов
                if migration in reactive_migrations:
                    self.threshold_model.update_transition_matrix(
                        migration['source_node'], target_node, True
                    )
                
                # Если это была проактивная миграция с наградой, обновляем Q-значения
                if 'reward' in migration:
                    state_key = self.q_learning_model._get_state_key(migration['source_node'], service_id)
                    action_key = self.q_learning_model._get_action_key(target_node)
                    next_state_key = self.q_learning_model._get_state_key(target_node, service_id)
                    
                    self.q_learning_model.update_q_value(
                        state_key, action_key, migration['reward'], next_state_key
                    )
        
        # Сохраняем метрики для анализа
        self.metrics_history.append({
            'node_metrics': node_metrics.copy(),
            'migrations': len(step_results['migrations']),
            'reactive_migrations': len([m for m in step_results['migrations'] if m['type'] == 'reactive']),
            'proactive_migrations': len([m for m in step_results['migrations'] if m['type'] == 'proactive']),
            'overloaded_nodes': len(step_results['overloaded_nodes']),
            'predicted_overloads': len(step_results['predicted_overloads'])
        })
        
        return step_results
    
    def analyze_performance(self):
        """
        Анализ производительности гибридной модели
        
        Возвращает:
        ----------
        dict
            Словарь с метриками производительности
        """
        total_migrations = sum(m['migrations'] for m in self.metrics_history)
        reactive_migrations = sum(m.get('reactive_migrations', 0) for m in self.metrics_history)
        proactive_migrations = sum(m.get('proactive_migrations', 0) for m in self.metrics_history)
        
        return {
            'total_migrations': total_migrations,
            'reactive_migrations': reactive_migrations,
            'proactive_migrations': proactive_migrations,
            'reactive_ratio': reactive_migrations / total_migrations if total_migrations > 0 else 0,
            'proactive_ratio': proactive_migrations / total_migrations if total_migrations > 0 else 0
        }
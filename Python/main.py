import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_scenarios import NetworkSimulator, run_all_scenarios
from visualization import visualize_all_scenarios, create_comparative_table

# Импорт моделей
from threshold_model import ThresholdMarkovModel
from qlearning_model import QLearningModel
from hybrid_model import HybridModel

# Настройка matplotlib для поддержки русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'

# Константы
NUM_NODES = 10
NUM_SERVICES = 20
SIMULATION_TIME = 1000

def initialize_models():
    """Инициализация моделей миграции"""
    # Создание простых представлений узлов и сервисов для моделей
    nodes = [{'id': i} for i in range(NUM_NODES)]
    services = [{'id': i} for i in range(NUM_SERVICES)]
    
    # Инициализация моделей с одинаковыми параметрами
    threshold_model = ThresholdMarkovModel(nodes, services, cpu_threshold=0.8, memory_threshold=0.8)
    qlearning_model = QLearningModel(nodes, services, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)
    hybrid_model = HybridModel(
        nodes, services, 
        cpu_threshold=0.8, memory_threshold=0.8,
        learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1
    )
    
    models = {
        'threshold': threshold_model,
        'qlearning': qlearning_model,
        'hybrid': hybrid_model
    }
    
    return models

def main():
    """Основная функция тестирования"""
    print("Инициализация моделей миграции...")
    models = initialize_models()
    
    print("Запуск тестирования на всех сценариях...")
    results = run_all_scenarios(models)
    
    print("Визуализация результатов...")
    visualize_all_scenarios()
    
    print("Создание сравнительной таблицы...")
    comparative_table = create_comparative_table()
    
    print("Тестирование завершено. Результаты сохранены в файлы CSV и PNG.")
    print("\nСравнительная таблица результатов:")
    print(comparative_table)

if __name__ == "__main__":
    main()
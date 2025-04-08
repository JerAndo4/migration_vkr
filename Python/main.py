import os
import time
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel
from hybrid_model import HybridModel
from test_scenarios import run_scenarios, ensure_directories
import plot_results



def diagnose_environment():
    """Проверка окружения на наличие всех необходимых модулей и файлов"""
    print("Диагностика окружения...")
    
    # Проверка наличия основных модулей
    required_modules = [
        ('threshold_model.py', ThresholdModel),
        ('qlearning_model.py', QLearningModel),
        ('hybrid_model.py', HybridModel),
        ('test_scenarios.py', run_scenarios),
        ('plot_results.py', plot_results)
    ]
    
    all_ok = True
    for module_file, module_object in required_modules:
        if os.path.exists(module_file):
            print(f"✓ Модуль {module_file} найден")
        else:
            print(f"✗ Модуль {module_file} не найден в текущей директории")
            all_ok = False
        
        if module_object is not None:
            print(f"✓ Класс/функция из {module_file} импортирован успешно")
        else:
            print(f"✗ Ошибка импорта из {module_file}")
            all_ok = False
    
    # Проверка доступности директорий
    for directory in ['results', 'plots']:
        try:
            os.makedirs(directory, exist_ok=True)
            # Проверка возможности записи
            test_file = os.path.join(directory, 'test_file.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✓ Директория {directory} доступна для записи")
        except Exception as e:
            print(f"✗ Проблема с директорией {directory}: {e}")
            all_ok = False
    
    return all_ok

def main():
    
    # Подготовка окружения
    ensure_directories()

    
    # Замеряем время выполнения
    start_time = time.time()
    
    # Запускаем все сценарии для всех моделей
    print("Начало моделирования всех сценариев...")
    results = run_scenarios()
    
    # Завершение первого этапа
    model_time = time.time() - start_time
    print(f"\nМоделирование завершено за {model_time:.2f} секунд.")
    
    # Генерируем графики и отчеты
    print("\nГенерация графиков и визуализаций...")
    plot_start_time = time.time()
    plot_results.main()
    plot_time = time.time() - plot_start_time
    
    # Выводим общее время выполнения
    total_time = time.time() - start_time
    print(f"\nВсе задачи выполнены за {total_time:.2f} секунд.")
    print(f"  - Моделирование: {model_time:.2f} секунд")
    print(f"  - Визуализация: {plot_time:.2f} секунд")
    
    print("\nРезультаты моделирования:")
    print(f" - CSV-файлы сохранены в директории 'results/'")
    print(f" - Графики сохранены в директории 'plots/'")
    
    print("\nДля детального анализа результатов рекомендуется изучить:")
    print(" - График сравнения миграций: plots/all_models_migrations.png")
    print(" - Графики задержки и джиттера для сценариев: plots/[scenario]_latency_jitter.png")
    print(" - Графики энергопотребления для сценариев: plots/[scenario]_energy.png")
    print(" - Сводную таблицу всех показателей: results/comparative_results.csv")
    print("\nЗавершение работы.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрерывание пользователем. Завершение работы.")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка при выполнении: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
import os
import time
import sys
from threshold_model import ThresholdModel
from qlearning_model import QLearningModel
from hybrid_model import HybridModel
from test_scenarios import run_scenarios
import plot_results

def ensure_directories():
    """Создает необходимые директории для результатов и графиков"""
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print("Директории для вывода результатов созданы.")

def print_header():
    """Выводит информационное сообщение о запуске симуляции"""
    print("=" * 80)
    print(" АВТОМАТИЗИРОВАННЫЕ МОДЕЛИ МИГРАЦИИ В СЕТЯХ НОВОГО ПОКОЛЕНИЯ")
    print(" Моделирование и анализ производительности")
    print("=" * 80)
    print("\n")

def main():
    """Основная функция запуска всего процесса моделирования и анализа"""
    print_header()
    
    # Подготовка окружения
    ensure_directories()
    
    # Замеряем время выполнения
    start_time = time.time()
    
    # Запускаем все сценарии для всех моделей
    print("Начало моделирования всех сценариев...")
    run_scenarios()
    
    # Генерируем графики и отчеты
    print("\nГенерация графиков и визуализаций...")
    plot_results.main()
    
    # Выводим общее время выполнения
    elapsed_time = time.time() - start_time
    print(f"\nВсе задачи выполнены за {elapsed_time:.2f} секунд.")
    
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
        sys.exit(1)
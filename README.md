# Модели миграции для сетей 5G/6G

## Обзор проекта

Этот проект реализует и оценивает три автоматизированные модели миграции для управления ресурсами в сетях нового поколения 5G/6G. Система моделирует поведение сети при различных сценариях нагрузки и измеряет ключевые показатели эффективности для сравнения эффективности различных подходов к миграции.

## Структура файлов

### Файлы исходного кода в каталоге Python

- `threshold_model.py` - Реализация пороговой модели с марковскими процессами
- `qlearning_model.py` - Реализация проактивной модели на основе Q-Learning
- `hybrid_model.py` - Реализация гибридной модели, объединяющей оба подхода
- `test_scenarios.py` - Среда моделирования сети с различными сценариями нагрузки
- `visualization.py` - Код для визуализации и анализа результатов моделирования
- `main.py` - Основной скрипт для запуска всех симуляций и генерации результатов

### Выходные файлы

После запуска моделирования будут сгенерированы следующие файлы:

#### CSV-файлы с данными в каталоге Results
- `critical_latency.csv`, `standard_latency.csv`, `mixed_latency.csv`, `dynamic_latency.csv` - Метрики задержки для каждого сценария
- `critical_jitter.csv`, `standard_jitter.csv`, `mixed_jitter.csv`, `dynamic_jitter.csv` - Метрики джиттера для каждого сценария
- `critical_energy.csv`, `standard_energy.csv`, `mixed_energy.csv`, `dynamic_energy.csv` - Метрики энергопотребления для каждого сценария
- `critical_migrations.csv`, `standard_migrations.csv`, `mixed_migrations.csv`, `dynamic_migrations.csv` - Количество миграций для каждого сценария
- `comparative_results.csv` - Комплексная таблица, сравнивающая все модели по всем сценариям

#### Файлы визуализации в каталоге Results
- `critical_results.png`, `standard_results.png`, `mixed_results.png`, `dynamic_results.png` - Графики производительности для каждого сценария
- `critical_migrations.png`, `standard_migrations.png`, `mixed_migrations.png`, `dynamic_migrations.png` - Визуализация количества миграций
- `summary_results.png` - Сводное сравнение ключевых метрик по всем сценариям
- `summary_migrations.png` - Сводное сравнение количества миграций по всем сценариям

Процесс моделирования:
1. Инициализирует три модели миграции с идентичными параметрами
2. Запускает каждую модель через четыре тестовых сценария
3. Генерирует CSV-файлы с данными о производительности
4. Создает визуализации, сравнивающие модели
5. Выводит комплексную сравнительную таблицу

## Модели миграции

### Пороговая модель с марковскими процессами
Реактивный подход, инициирующий миграции при превышении использованием ресурсов заданных пороговых значений. Использует марковские процессы для интеллектуального выбора целевых узлов на основе исторических паттернов миграции.

### Модель Q-Learning
Проактивный подход, использующий обучение с подкреплением для прогнозирования будущей нагрузки и инициирования превентивных миграций до возникновения перегрузок. Модель обучается оптимальным стратегиям миграции через опыт.

### Гибридная модель
Объединяет реактивный и проактивный подходы, используя пороговую модель для немедленного реагирования на перегрузки и Q-Learning для оптимизации распределения ресурсов в периоды стабильной работы.

## Сценарии моделирования

Система оценивает каждую модель в четырех различных сценариях:

1. **Критический сценарий**: Высокая нагрузка на критические сервисы, проверка способности моделей справляться с экстремальными условиями
2. **Стандартный сценарий**: Умеренная нагрузка с небольшими колебаниями, представляющая типичные условия эксплуатации
3. **Смешанный сценарий**: Чередующиеся периоды стабильной и высокой нагрузки, проверка адаптивности к изменяющимся условиям
4. **Динамический сценарий**: Сложные паттерны нагрузки с трендами, сезонностью и случайными компонентами, имитирующие реальные условия эксплуатации

## Показатели эффективности

Моделирование измеряет и анализирует следующие ключевые метрики:

- **Задержка**: Взвешенная средняя задержка по всем сервисам (меньше - лучше)
- **Джиттер**: Вариация задержки во времени (меньше - лучше)
- **Энергопотребление**: Общее энергопотребление всех узлов (меньше - лучше)
- **Количество миграций**: Количество выполненных миграций сервисов (обычно меньше - лучше)

Для гибридной модели миграции дополнительно классифицируются как реактивные или проактивные для более глубокого анализа поведения модели.

## Руководство по визуализации

- Линейные графики показывают изменение метрик (задержка, джиттер, энергопотребление) во времени для каждой модели
- Столбчатые диаграммы сравнивают количество миграций между моделями
- Сводные графики предоставляют обзор средней производительности по всем сценариям
- Сравнительная таблица в формате CSV предлагает детальный статистический анализ для дальнейшего изучения

## Дополнительные примечания

Параметры моделирования (количество узлов, сервисов и продолжительность моделирования) можно изменить в файле `main.py` при необходимости:

```python
NUM_NODES = 10      # По умолчанию: 10 узлов
NUM_SERVICES = 20   # По умолчанию: 20 сервисов
SIMULATION_TIME = 1000  # По умолчанию: 1000 временных тактов
```

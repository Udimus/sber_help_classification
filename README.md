# Тестовое задание на позицию DS'а

## Описание датасета

В этом задании предлагается классифицировать сообщения на предмет того,
из какого чата они были взяты. Датасет представляет собой сообщения
участников двух публичных чатов:
* Чат по Python (label=0)
* Чат по Data Science (label=1)

## Описание задачи

Ваша задача:
1. Сделать предобработку текстов
2. Предложить модель классификации, метрики.
3. Объяснить их выбор (достоинства/недостатки). 
4. Сделать выводы по работе модели(ей) (достоинства/недостатки),
а также проблемы, с которыми вы или ваша модель столкнулась
5. Развернуть модель в качестве REST API в докер-контейнере
Тестовое задание в Сбер.


## Репозиторий

### Ноутбуки
* DataAnalysis -- исследование данных.
* Baseline -- выбор метрик, построение базового решения.
* Catboost -- итоговое решение с помощью catboost.
* Embeddings -- попытка применить векторные представления и сетки.
* TestAPI -- ноутбук с проверкой сервиса (и примерами запросов).

### Модуль
Файлы с классами для предобработки данных и модели.

### Сервис
Сервис, в который можно обратиться по API.


## Сборка

```docker build -t test_help_sber_solution .```

## Запуск

```docker run -p 5000:5000 test_help_sber_solution```

# Проект спринта 10 : Обучение с учителем

## Задача

Нужно построить модель машинного обучения для прогнозирования - уйдёт клиент из банка в ближайшее время или нет. 

## Выводы

Провели очистку данных от пропусков, удалили ненужные фичи, тектовые - перекодировали методом OHE. Также провели перебалансировку классов через upsampling. После - перебрали несколько вариантов моделей с подбором гиперпараметров. Результат: добились цели по метрике F1 с моделью случайного леса.


## Используемые библиотеки

pandas
sklearn
matplotlib

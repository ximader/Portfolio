# Проект спринта 15 : Прогнозирование заказов такси

## Задача

Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Нужно построить модель для такого предсказания.

Значение метрики RMSE на тестовой выборке должно быть не больше 48.

## Вывод

После разделения выборки на составляющие через seasonal_decompose() заметили сезонность в рамках суток. Наиболее загруженные часы - ночные, наименее загруженные - утренние. Также имеет место четкий тренд на возрастание кол-ва заказов/час, что, вероятно связано с характером данных (аэоропорты) - в летние месяцы туристический поток возрастает, что и увеличивает нагрузку на такси.  

Обучили ряд моделей, где ансамбли показали себя лучше всего. Наилучший результат у LightBGM  45.6.


## Используемые библиотеки

pandas
sklearn
matplotlib
statsmodels

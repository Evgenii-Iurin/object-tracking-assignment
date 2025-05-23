# Object Tracking Home assignment

### Установка зависимостей
```
pip install -r requirements.txt
```
Or more safe

```
poetry install
poetry shell
```

Note : poetry uses `python3.11` or `python3.12`


### Запуск сервера
Или настройте запуск файла fastapi_server.py как приведено на скриншоте ниже 
![fastapi.png](info/fastapi.png)

или командой в терминале
```
python3 -m uvicorn fastapi_server:app --reload --port 8001 
```

Запуск веб сервера http://localhost:8000/
```
python3 -m http.server
```

### Постановка задачи

Реализуйте методы tracker_soft и tracker_strong в скрипте fastapi_server.py,
придумайте, обоснуйте и реализуйте метод для оценки качества разработанных трекеров.
Сравните результаты tracker_soft и tracker_strong для 5, 10, 20 объектов и различных 
значений random_range и bb_skip_percent
(информацию о генерации данных читай в пункте "Тестирование"). Напишите отчёт. 
В отчете необходимо в свободном стиле привести описание методов tracker_soft, 
tracker_strong, метода оценки качества трекеров, привести сравнительную таблицу 
реализованных трекеров, сделать вывод.  
Бонусом можете выписать найденные баги в текущем проекте.

### Тестирование
Для тестирования можно воспользоваться скриптом create_track.py. Скрипт генерирует
информацию об объектах и их треках. Скопируйте вывод в новый скрипт track_n.py и
скорректируйте импорт в fastapi_server.py
```
from track_n import track_data, country_balls_amount
```
Что стоит менять в скрипте create_track.py:  
**tracks_amount**: количество объектов  
**random_range**: на сколько пикселей рамка объектов может ложно смещаться (эмуляция не идеальной детекции)  
**bb_skip_percent**: с какой вероятностью объект на фрейме может быть не найдет детектором  
from flask import Flask, render_template, request, url_for, send_from_directory
import pandas as pd
import numpy as np
import os
from annoy import AnnoyIndex # Для ускоренного поиска
import json

app = Flask(__name__)

# --- Константы и конфигурация ---
CSV_PATH = os.path.join(app.root_path, 'data', 'products.csv')
EMBEDDING_DIM = 128  # Размерность эмбеддинга (подогнать под вашу модель)
N_TREES = 100        # Количество деревьев для Annoy (больше - точнее, но дольше строится)
N_RESULTS = 10       # Количество результатов поиска для отображения
ANNOY_INDEX_PATH = os.path.join(app.root_path, 'data', 'embeddings.ann')
PRODUCT_DATA_PATH = os.path.join(app.root_path, 'data', 'products_with_embeddings.json') # Для сохранения данных с dummy эмбеддингами

# Глобальные переменные для хранения данных
PRODUCTS_DF = None
ANNOY_INDEX = None
PRODUCT_ID_MAP = {} # Словарь для быстрого доступа к товару по ID

def generate_and_save_dummy_embeddings():
    """
    Генерирует случайные эмбеддинги для каждого товара из CSV
    и сохраняет их вместе с данными о товарах в JSON,
    а также строит и сохраняет Annoy индекс.
    Эта функция вызывается только если Annoy индекс не найден.
    """
    global PRODUCTS_DF, ANNOY_INDEX, PRODUCT_ID_MAP

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Ошибка: CSV файл не найден по пути {CSV_PATH}. Пожалуйста, убедитесь, что 'products.csv' существует.")
    
    PRODUCTS_DF = pd.read_csv(CSV_PATH)
    # Заполняем NaN в колонках изображений пустыми строками
    image_path_cols = [col for col in PRODUCTS_DF.columns if 'local_image_path_' in col]
    PRODUCTS_DF[image_path_cols] = PRODUCTS_DF[image_path_cols].fillna('')

    print(f"Загружено {len(PRODUCTS_DF)} товаров из CSV.")

    # Генерируем случайные эмбеддинги
    PRODUCTS_DF['embedding'] = [np.random.rand(EMBEDDING_DIM).tolist() for _ in range(len(PRODUCTS_DF))]

    # Сохраняем расширенные данные о товарах (с эмбеддингами) в JSON для удобства
    os.makedirs(os.path.dirname(PRODUCT_DATA_PATH), exist_ok=True) # Убедимся, что папка data существует
    PRODUCTS_DF.to_json(PRODUCT_DATA_PATH, orient='records', indent=4)
    print(f"Данные товаров с эмбеддингами сохранены в {PRODUCT_DATA_PATH}")

    # Строим Annoy индекс
    ANNOY_INDEX = AnnoyIndex(EMBEDDING_DIM, 'angular') # 'angular' для косинусного сходства
    for i, row in PRODUCTS_DF.iterrows():
        ANNOY_INDEX.add_item(i, row['embedding']) # Добавляем индекс Annoy и эмбеддинг
    ANNOY_INDEX.build(N_TREES)
    os.makedirs(os.path.dirname(ANNOY_INDEX_PATH), exist_ok=True) # Убедимся, что папка data существует
    ANNOY_INDEX.save(ANNOY_INDEX_PATH)
    print(f"Annoy индекс построен и сохранен в {ANNOY_INDEX_PATH}")

    # Создаем карту ID для быстрого доступа
    PRODUCT_ID_MAP = {str(row.name): row.to_dict() for idx, row in PRODUCTS_DF.iterrows()}


def load_data():
    """
    Загружает данные о товарах и Annoy индекс.
    Если индекс не существует, генерирует эмбеддинги и строит его.
    """
    global PRODUCTS_DF, ANNOY_INDEX, PRODUCT_ID_MAP

    if os.path.exists(ANNOY_INDEX_PATH) and os.path.exists(PRODUCT_DATA_PATH):
        print("Загрузка данных и Annoy индекса из файлов...")
        PRODUCTS_DF = pd.read_json(PRODUCT_DATA_PATH)
        ANNOY_INDEX = AnnoyIndex(EMBEDDING_DIM, 'angular')
        ANNOY_INDEX.load(ANNOY_INDEX_PATH)
        PRODUCT_ID_MAP = {str(row.name): row.to_dict() for idx, row in PRODUCTS_DF.iterrows()}
        print(f"Загружено {len(PRODUCTS_DF)} товаров и Annoy индекс.")
    else:
        print("Annoy индекс или данные не найдены. Генерируем эмбеддинги и строим индекс...")
        generate_and_save_dummy_embeddings()

def get_query_embedding(text_query):
    """
    Заглушка для получения эмбеддинга текстового запроса.
    В реальном проекте здесь будет вызов вашей модели.
    """
    # Для демо: генерируем случайный эмбеддинг
    return np.random.rand(EMBEDDING_DIM)

@app.route('/', methods=['GET', 'POST'])
def index():
    search_results = []
    query_text = ""
    if request.method == 'POST':
        query_text = request.form.get('query')
        if query_text:
            print(f"Поиск по запросу: '{query_text}'")
            query_embedding = get_query_embedding(query_text)

            neighbor_indices, distances = ANNOY_INDEX.get_nns_by_vector(
                query_embedding, N_RESULTS, include_distances=True
            )

            for i, idx in enumerate(neighbor_indices):
                product_data = PRODUCTS_DF.iloc[idx].to_dict()
                similarity = 1 - (distances[i] / np.sqrt(2)) 
                if similarity < 0: similarity = 0 
                
                product_result = {
                    "id": str(idx), 
                    "name": product_data.get('name', 'Нет названия'),
                    "description": product_data.get('description', 'Нет описания'),
                    "price": product_data.get('price', 'Н/Д'),
                    "url": product_data.get('url', '#'),
                    "similarity": f"{similarity:.4f}",
                    "image_paths": []
                }
                
                # Собираем все пути к изображениям и проверяем их существование
                found_images = False
                for col in [c for c in PRODUCTS_DF.columns if 'local_image_path_' in c]:
                    relative_path = product_data.get(col)
                    if relative_path:
                        full_img_path = os.path.join(app.root_path, 'static', relative_path)
                        if os.path.exists(full_img_path):
                            product_result['image_paths'].append(relative_path)
                            found_images = True
                        else:
                            print(f"Внимание: Изображение не найдено по пути: {full_img_path} для товара ID {product_result['id']}")
                
                if not found_images:
                    # Если ни одной картинки не найдено, мы не можем просто добавить заглушку,
                    # т.к. вы не хотите автоматических заглушек.
                    # Вместо этого, можно пропустить товар или показать ошибку/предупреждение
                    print(f"Ошибка: Для товара ID {product_result['id']} не найдено ни одного изображения.")
                    continue # Пропускаем товар без изображений

                search_results.append(product_result)
            print(f"Найдено {len(search_results)} результатов.")

    return render_template('index.html', search_results=search_results, query_text=query_text)

@app.route('/product/<product_id>')
def product_detail(product_id):
    try:
        idx = int(product_id)
    except ValueError:
        return "Неверный ID товара", 400

    product_data_dict = PRODUCT_ID_MAP.get(product_id)
    if product_data_dict:
        image_paths = []
        found_images = False
        for col in [c for c in PRODUCTS_DF.columns if 'local_image_path_' in c]:
            relative_path = product_data_dict.get(col)
            if relative_path:
                full_img_path = os.path.join(app.root_path, 'static', relative_path)
                if os.path.exists(full_img_path):
                    image_paths.append(relative_path)
                    found_images = True
                else:
                    print(f"Внимание: Изображение не найдено по пути: {full_img_path} для товара ID {product_id} на странице деталей.")

        if not found_images and image_paths: # Если пути были, но файлы не найдены
             raise FileNotFoundError(f"Ошибка: Ни одного изображения не найдено для товара ID {product_id}. Проверьте пути: {image_paths}")
        elif not image_paths: # Если путей вообще не было в CSV
            raise ValueError(f"Ошибка: В CSV для товара ID {product_id} не указаны пути к изображениям (local_image_path_X).")


        product = {
            "id": product_id,
            "name": product_data_dict.get('name', 'Нет названия'),
            "description": product_data_dict.get('description', 'Нет описания'),
            "price": product_data_dict.get('price', 'Н/Д'),
            "url": product_data_dict.get('url', '#'),
            "category": product_data_dict.get('category', 'Н/Д'),
            "design_size": product_data_dict.get('Размер дизайна, мм', 'Н/Д'),
            "stitch_count": product_data_dict.get('Количество стежков', 'Н/Д'),
            "color_count": product_data_dict.get('Количество цветов', 'Н/Д'),
            "file_formats": product_data_dict.get('Форматы файлов', 'Н/Д'),
            "image_paths": image_paths
        }
        return render_template('product.html', product=product)
    return "Товар не найден", 404


@app.route('/static/<path:filename>')
def custom_static(filename):
    """
    Обслуживает статические файлы из папки static.
    В случае отсутствия файла выдаст 404, а не пустую заглушку.
    """
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    try:
        load_data()
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАГРУЗКЕ ДАННЫХ: {e}")
        print("Приложение не может быть запущено. Убедитесь в наличии products.csv и корректности путей к изображениям.")
        exit(1) # Выход из приложения при критической ошибке

    app.run(debug=True)
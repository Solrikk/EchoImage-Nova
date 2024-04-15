import aiohttp
import asyncio
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from skimage.metrics import structural_similarity as ssim

from database import load_db

model = ResNet50(weights='imagenet')
model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)


def preprocess_image(img_path):
  img = keras_image.load_img(img_path, target_size=(224, 224))
  img_array = keras_image.img_to_array(img)
  img_array_expanded = np.expand_dims(img_array, axis=0)
  return preprocess_input(img_array_expanded)


async def extract_features(img_path, model=model):
  img = preprocess_image(img_path)
  features = model.predict(img)
  return features.flatten()


async def compare_features(feature1, feature2):
  similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) *
                                             np.linalg.norm(feature2))
  return similarity


async def find_similar_images(file_path):
  db_data = load_db()
  target_image_features = await extract_features(file_path)
  async with aiohttp.ClientSession() as session:
    tasks = []
    for entry in db_data:
      task = asyncio.ensure_future(extract_features(entry.get('file_path',
                                                              '')))
      tasks.append(task)
    all_features = await asyncio.gather(*tasks)
    similarities = [
        await compare_features(target_image_features, current_features)
        for current_features in all_features
    ]
    results = zip(similarities, db_data)
    valid_results = filter(lambda x: x[0] > 0.5, results)
    sorted_results = sorted(valid_results, key=lambda x: x[0],
                            reverse=True)[:5]
  return [
      result[1].get('file_path', 'Missing file path')
      for result in sorted_results
  ]

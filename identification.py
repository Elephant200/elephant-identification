from PIL import Image
from pprint import pprint
from urllib.request import urlopen

import numpy as np
import pandas as pd
import os
import time
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

from wildlife_datasets.datasets import ELPephants
from wildlife_tools.data.dataset import WildlifeDataset, FeatureDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier, TopkClassifier

if __name__ == "__main__":
  if os.path.exists("train_cache/metadata.csv") and not input("Reload metadata? (y/n): ") == "y":
    print("Using cached metadata...")
    metadata = pd.read_csv("train_cache/metadata.csv")
  else:
    print("Loading metadata...")
    metadata = ELPephants(root='images/ELPephants').df
    metadata.to_csv("train_cache/metadata.csv", index=False)
  
  transform = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

  dataset_database = WildlifeDataset(metadata.iloc[100:,:], "images/ELPephants", transform=transform)
  dataset_query = WildlifeDataset(metadata.iloc[:100,:], "images/ELPephants", transform=transform)

  print("Loading model...")
  model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-224", num_classes=0, pretrained=True)
  if os.path.exists("train_cache/query.pkl") and os.path.exists("train_cache/database.pkl") and not input("Reload features? (y/n): ") == "y":
    print("Using cached features...")
    query = FeatureDataset.from_file("train_cache/query.pkl")
    database = FeatureDataset.from_file("train_cache/database.pkl")
  else:
    print("Extracting features...")
    extractor = DeepFeatures(model, num_workers=8)
    query, database = extractor(dataset_query), extractor(dataset_database)

    print("Saving features...")
    query.save("train_cache/query.pkl")
    database.save("train_cache/database.pkl")
  
  print("Calculating similarity...")
  similarity_function = CosineSimilarity()
  similarity = similarity_function(query, database)

  print("Classifying...")
  classifier = TopkClassifier(database_labels=dataset_database.labels_string, k=1)
  predictions = classifier(similarity)
  accuracy = np.mean(dataset_query.labels_string == predictions)

  print(f"Accuracy: {accuracy}")

  # img = Image.open(urlopen(
  #     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
  # ))



  # # Load model from Huggingface Hub
  # model = timm.create_model(
  #     "hf-hub:BVRA/MegaDescriptor-L-384",
  #     pretrained=True)

  # model = model.eval()

  # # Load expected image transformations
  # transforms = T.Compose([T.Resize(224),
  # T.ToTensor(),
  # T.Normalize(
  #     [0.5, 0.5, 0.5],
  #     [0.5, 0.5, 0.5])])

  # # Load/feed-forward image to MegaDescriptor
  # image = Image.open("./test_image.png")
  # output = model(transforms(image).unsqueeze(0))



  # # Load database of image features
  # database = FeatureDatabase.from_file(
  # "database_file"
  # )
  # # Extract features from query image
  # image = Image.open("./query_image.png")
  # query = model(transforms(image).unsqueeze(0))
  # # Find nearest match in database
  # matcher = KnnMatcher(database)
  # matcher([query])
  # print(matcher([query]))
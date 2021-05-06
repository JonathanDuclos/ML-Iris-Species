#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:35:50 2021

@author: slammernet

Objetivo: Proprio codigo de ML para classificacao de Iris (plantas/flores) segundo suas caracteristicas
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


csvPath = "/home/slammernet/Downloads/archive/Iris.csv"
csvTestPath = "/home/slammernet/Downloads/archive/Iris_test.csv"
csv = pd.read_csv(csvPath)
csv_test = pd.read_csv(csvTestPath)

columns = []
outputs = []

for col in csv.columns:
    columns.append(col)
    

for line in csv['Species']:
    if line not in outputs:
        outputs.append(line)
    
    
csv['Species'] = pd.Categorical(csv['Species'])
csv['Species'] = csv.Species.cat.codes
csv_test['Species'] = pd.Categorical(csv_test['Species'])
csv_test['Species'] = csv_test.Species.cat.codes

csv_labels = csv.pop('Species')
csv_test_labels = csv_test.pop('Species')
csv.pop('Id')
csv_test.pop('Id')

print("CSV",csv)
print("\n\n\n CSV TEST", csv_test)

dataset_origin = tf.data.Dataset.from_tensor_slices((csv, csv_labels))
dataset_test = tf.data.Dataset.from_tensor_slices((csv_test, csv_test_labels))

train_dataset = dataset_origin.shuffle(len(csv)).batch(1)
print("TrainDataset",train_dataset)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=150)
loss, acc = model.evaluate(csv_test, csv_test_labels, verbose=2)
prediction = model.predict(csv_test)
print(" Precisao: ", acc)
print(prediction, "\n\n Alvo: ", csv_test_labels)

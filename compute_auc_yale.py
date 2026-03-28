import os
import random
import numpy as np
import cv2
from itertools import combinations

# -------- SETTINGS -------- #
DATASET = "./cropped"
SCORES_FILE = "./cropped/yale_scores.txt"

# -------- LOAD SCORES -------- #
scores = {}
with open(SCORES_FILE) as f:
    for line in f:
        path, s = line.strip().split()
        scores[path] = float(s)

# -------- LOAD DATA -------- #
data = {}
for person in os.listdir(DATASET):
    p_path = os.path.join(DATASET, person)
    if not os.path.isdir(p_path):
        continue
    imgs = [os.path.join(p_path, i) for i in os.listdir(p_path)]
    if len(imgs) > 1:
        data[person] = imgs

# -------- GENERATE PAIRS -------- #
pairs = []

# positive pairs
for person, imgs in data.items():
    for a, b in combinations(imgs, 2):
        pairs.append((a, b, 1))

# negative pairs
persons = list(data.keys())
for _ in range(len(pairs)):
    p1, p2 = random.sample(persons, 2)
    a = random.choice(data[p1])
    b = random.choice(data[p2])
    pairs.append((a, b, 0))

print("Total pairs:", len(pairs))

# -------- SIMPLE FR (embedding) -------- #
def get_embedding(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (112, 112))
    return img.flatten() / 255.0

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# -------- PRECOMPUTE EMBEDDINGS -------- #
embeddings = {}

for person, imgs in data.items():
    for img in imgs:
        embeddings[img] = get_embedding(img)

print("Embeddings computed:", len(embeddings))

# -------- COMPUTE RESULTS -------- #

results = []

for img1, img2, label in pairs:
    e1 = embeddings[img1]
    e2 = embeddings[img2]

    sim = cosine(e1, e2)
    q = min(scores[img1], scores[img2])

    results.append((sim, label, q))

    if len(results) % 100000 == 0:
        print("pairs added:", len(results))

print("Results computed:", len(results))

# -------- FNMR @ FMR -------- #
def fnmr_at_fmr(sims, labels, fmr_target=1e-3):
    pos = [s for s, l in zip(sims, labels) if l == 1]
    neg = [s for s, l in zip(sims, labels) if l == 0]

    neg = sorted(neg)
    # idx = int((1 - fmr_target) * len(neg))
    idx = max(0, min(len(neg)-1, int((1 - fmr_target) * len(neg))))
    threshold = neg[idx]

    fn = sum(1 for s in pos if s < threshold)
    return fn / len(pos)

# -------- AUC -------- #
results = sorted(results, key=lambda x: x[2])  # sort by quality

rejection_rates = np.linspace(0, 1.0, 20)
fnmr_list = []
valid_r = []

results = sorted(results, key=lambda x: x[2])

for r in rejection_rates:
    cut = int(r * len(results))

    if cut >= len(results):   # avoid empty subset
        break

    subset = results[cut:]

    sims = [x[0] for x in subset]
    labels = [x[1] for x in subset]

    fnmr = fnmr_at_fmr(sims, labels)

    valid_r.append(r)
    fnmr_list.append(fnmr)

print("FNMR computed:", len(fnmr_list))

auc = np.trapz(fnmr_list, valid_r)

print("\nFINAL RESULT")
print("AUC @ FMR=1e-3 =", auc)

import matplotlib.pyplot as plt

rejection_rates = np.linspace(0, 0.5, 20)
fnmr_list = []

results = sorted(results, key=lambda x: x[2])

for r in rejection_rates:
    cut = int(r * len(results))
    subset = results[cut:]

    sims = [x[0] for x in subset]
    labels = [x[1] for x in subset]

    fnmr = fnmr_at_fmr(sims, labels)
    fnmr_list.append(fnmr)

# -------- PLOT -------- #
plt.figure()
plt.plot(rejection_rates, fnmr_list)
plt.xlabel("Rejection Rate")
plt.ylabel("FNMR")
plt.title("Error vs Discard Curve (EDC)")
plt.grid()
plt.show()

plt.savefig("edc_curve.png")
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


output_dir = "evaluation/plots"
os.makedirs(output_dir, exist_ok=True)


with open("evaluation/results.json", "r") as f:
    data = json.load(f)


def get_metric(entry, metric="time_seconds"):
    if metric == "f1":
        return entry["classification_report"]["weighted avg"]["f1-score"]
    if metric == "accuracy":
        return entry["accuracy"]
    return entry[metric]


workers = [2, 3, 4]
models = ["Gradient Boosted Trees", "Random Forest", "Multilayer Perceptron"]

colors = {
    "Gradient Boosted Trees": "#1f77b4",  
    "Random Forest": "#2ca02c",          
    "Multilayer Perceptron": "#d62728"   
}

time_data = {m: [] for m in models}

for w in workers:
    key = f"{w}_workers"
    if key in data and "100%" in data[key]:
        entries = data[key]["100%"]
        
        entry_map = {e["model"]: e for e in entries}
        for m in models:
            if m in entry_map:
                time_data[m].append(entry_map[m]["time_seconds"])
            else:
                time_data[m].append(None)
    else:
        for m in models:
            time_data[m].append(None)

plt.figure(figsize=(10, 6))
for model, times in time_data.items():
    if any(t is not None for t in times):
        plt.plot(workers, times, marker='o', label=model, color=colors[model])

plt.title("Computation Time vs Number of Workers (100% Data)")
plt.xlabel("Number of Workers")
plt.ylabel("Time (seconds)")
plt.xticks(workers)
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/plot1_time_vs_workers.png")
plt.close()


percentages = ["25%", "50%", "75%", "100%"]
pct_values = [25, 50, 75, 100]
acc_data = {m: [] for m in models}
entries_3w = data.get("3_workers", {})

for pct in percentages:
    if pct in entries_3w:
        entries = entries_3w[pct]
        entry_map = {e["model"]: e for e in entries}
        for m in models:
            if m in entry_map:
                acc_data[m].append(entry_map[m]["accuracy"])
            else:
                acc_data[m].append(0) 
    else:
        for m in models:
            acc_data[m].append(0)

x = np.arange(len(percentages))
width = 0.25
multiplier = 0

plt.figure(figsize=(10, 6))

for i, model in enumerate(models):
    offset = width * multiplier
    rects = plt.bar(x + offset, acc_data[model], width, label=model, color=colors[model])
    multiplier += 1

plt.title("Model Accuracy by Data Percentage (3 Workers)")
plt.xlabel("Data Percentage")
plt.ylabel("Accuracy")
plt.xticks(x + width, percentages)
plt.legend(loc='lower right')
plt.ylim(0, 1.0)
plt.grid(axis='y')
plt.savefig(f"{output_dir}/plot2_accuracy_vs_data.png")
plt.close()


time_data_pct = {m: [] for m in models}

for pct in percentages:
    if pct in entries_3w:
        entries = entries_3w[pct]
        entry_map = {e["model"]: e for e in entries}
        for m in models:
            if m in entry_map:
                time_data_pct[m].append(entry_map[m]["time_seconds"])
            else:
                time_data_pct[m].append(None)
    else:
        for m in models:
            time_data_pct[m].append(None)

plt.figure(figsize=(10, 6))
for model, times in time_data_pct.items():
    if any(t is not None for t in times):
        plt.plot(pct_values, times, marker='s', label=model, color=colors[model])

plt.title("Computation Time by Data Percentage (3 Workers)")
plt.xlabel("Data Percentage")
plt.ylabel("Time (seconds)")
plt.xticks(pct_values, percentages)
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/plot3_time_vs_data.png")
plt.close()


recall_vals = []

if "100%" in entries_3w:
    entries = entries_3w["100%"]
    entry_map = {e["model"]: e for e in entries}
    for m in models:
        if m in entry_map:
            recall_vals.append(entry_map[m]["classification_report"]["weighted avg"]["recall"])
        else:
            recall_vals.append(0)

plt.figure(figsize=(8, 6))
bars = plt.bar(models, recall_vals, color=[colors[m] for m in models])

plt.title("Recall Scores (Weighted Avg) on 100% Data (3 Workers)")
plt.ylabel("Recall Score")
plt.ylim(0, 1.0)
plt.grid(axis='y')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.savefig(f"{output_dir}/plot4_recall_scores.png")
plt.close()

print(f"Plots saved to {os.path.abspath(output_dir)}")

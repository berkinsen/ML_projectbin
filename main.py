import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, adjusted_rand_score, silhouette_score
)
seeds = [42, 10, 100]
classifier_configs = [
    ("KNN (K=1)", lambda: KNeighborsClassifier(n_neighbors=1), True),
    ("KNN (K=3)", lambda: KNeighborsClassifier(n_neighbors=3), True),
    ("KNN (K=5)", lambda: KNeighborsClassifier(n_neighbors=5), True),
    ("KNN (K=7)", lambda: KNeighborsClassifier(n_neighbors=7), True),
    ("KNN (K=9)", lambda: KNeighborsClassifier(n_neighbors=9), True),
    ("KNN (K=11)", lambda: KNeighborsClassifier(n_neighbors=11), True),
    
    ("SVM (Linear C=0.1)", lambda: SVC(kernel='linear', C=0.1, random_state=42), True),
    ("SVM (Linear C=1.0)", lambda: SVC(kernel='linear', C=1.0, random_state=42), True),
    ("SVM (Linear C=10.0)", lambda: SVC(kernel='linear', C=10.0, random_state=42), True),
    ("SVM (RBF C=0.1)", lambda: SVC(kernel='rbf', C=0.1, random_state=42), True),
    ("SVM (RBF C=1.0)", lambda: SVC(kernel='rbf', C=1.0, random_state=42), True),
    ("SVM (RBF C=10.0)", lambda: SVC(kernel='rbf', C=10.0, random_state=42), True),
    ("SVM (Poly d=3)", lambda: SVC(kernel='poly', degree=3, random_state=42), True),
   
    ("Decision Tree (depth=3)", lambda: DecisionTreeClassifier(max_depth=3, random_state=42), False),
    ("Decision Tree (depth=5)", lambda: DecisionTreeClassifier(max_depth=5, random_state=42), False),
    ("Decision Tree (depth=7)", lambda: DecisionTreeClassifier(max_depth=7, random_state=42), False),
    ("Decision Tree (unconstrained)", lambda: DecisionTreeClassifier(random_state=42), False),
    
    ("Random Forest (trees=10)", lambda: RandomForestClassifier(n_estimators=10, random_state=42), False),
    ("Random Forest (trees=50)", lambda: RandomForestClassifier(n_estimators=50, random_state=42), False),
    ("Random Forest (trees=100)", lambda: RandomForestClassifier(n_estimators=100, random_state=42), False),
    ("Random Forest (trees=200)", lambda: RandomForestClassifier(n_estimators=200, random_state=42), False),
    ("Random Forest (depth=5, trees=100)", lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), False)
]
datasets = {
    "wine": load_wine(),
    "cancer": load_breast_cancer()
}
all_classifier_results = []
all_kmeans_results = []
for ds_name, ds_data in datasets.items():
    X = pd.DataFrame(ds_data.data, columns=ds_data.feature_names)
    y = ds_data.target
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        for model_name, model_fn, use_scaled in classifier_configs:
            model = model_fn() 
            if use_scaled:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec = recall_score(y_test, preds, average='macro', zero_division=0)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    
            all_classifier_results.append({
                "Dataset": ds_name,
                "Seed": seed,
                "Model": model_name,
                "Scaling": "Yes" if use_scaled else "No",
                "Accuracy": acc,
                "Precision_Macro": prec,
                "Recall_Macro": rec,
                "F1_Macro": f1
            })

    n_classes = len(np.unique(y))
    k_values = [n_classes, n_classes + 1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for seed in seeds:
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            cluster_labels = km.fit_predict(X_scaled)
            
            ari = adjusted_rand_score(y, cluster_labels)
            sil = silhouette_score(X_scaled, cluster_labels)
            inertia = km.inertia_

            all_kmeans_results.append({
                "Dataset": ds_name,
                "Seed": seed,
                "K_Clusters": k,
                "ARI": ari,
                "Silhouette": sil,
                "Inertia": inertia
            })

df_clf = pd.DataFrame(all_classifier_results)
df_km = pd.DataFrame(all_kmeans_results)

print(f"--- EXPERIMENT SUMMARY ---")
print(f"Classifier Fits: {len(df_clf)}")
print(f"K-Means Fits: {len(df_km)}")
print(f"Total Fits: {len(df_clf) + len(df_km)}\n")

df_wine_clf = df_clf[df_clf["Dataset"] == "wine"]
df_wine_clf.to_csv("wine_classifier_all_runs.csv", index=False)
df_wine_summary = df_wine_clf.groupby(["Model", "Scaling"]).agg(
    Mean_Accuracy=("Accuracy", "mean"),
    Std_Accuracy=("Accuracy", "std"),
    Mean_Precision=("Precision_Macro", "mean"),
    Mean_Recall=("Recall_Macro", "mean"),
    Mean_F1=("F1_Macro", "mean")
).reset_index()
df_wine_summary.to_csv("wine_classifier_summary.csv", index=False)
df_cancer_clf = df_clf[df_clf["Dataset"] == "cancer"]
df_cancer_clf.to_csv("cancer_classifier_all_runs.csv", index=False)
df_cancer_summary = df_cancer_clf.groupby(["Model", "Scaling"]).agg(
    Mean_Accuracy=("Accuracy", "mean"),
    Std_Accuracy=("Accuracy", "std"),
    Mean_Precision=("Precision_Macro", "mean"),
    Mean_Recall=("Recall_Macro", "mean"),
    Mean_F1=("F1_Macro", "mean")
).reset_index()
df_cancer_summary.to_csv("cancer_classifier_summary.csv", index=False)
df_km.to_csv("kmeans_all_runs.csv", index=False)
df_km_summary = df_km.groupby(["Dataset", "K_Clusters"]).agg(
    Mean_ARI=("ARI", "mean"),
    Std_ARI=("ARI", "std"),
    Mean_Silhouette=("Silhouette", "mean"),
    Mean_Inertia=("Inertia", "mean")
).reset_index()
df_km_summary.to_csv("kmeans_summary.csv", index=False)

X_w, y_w = datasets["wine"].data, datasets["wine"].target
X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
cm_wine = confusion_matrix(y_te, rf.predict(X_te))

plt.figure(figsize=(6, 5))
sns.heatmap(cm_wine, annot=True, fmt='d', cmap='Blues', xticklabels=datasets["wine"].target_names, yticklabels=datasets["wine"].target_names)
plt.title("Wine Classification Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("wine_confusion_matrix.png", dpi=300)
plt.close()

X_c, y_c = datasets["cancer"].data, datasets["cancer"].target
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)
svm = SVC(kernel='linear', C=1.0, random_state=42).fit(X_tr_sc, y_tr)
cm_cancer = confusion_matrix(y_te, svm.predict(X_te_sc))

plt.figure(figsize=(6, 5))
sns.heatmap(cm_cancer, annot=True, fmt='d', cmap='Greens', xticklabels=datasets["cancer"].target_names, yticklabels=datasets["cancer"].target_names)
plt.title("Breast Cancer Confusion Matrix (Linear SVM)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("cancer_confusion_matrix.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_wine_summary, x="Model", y="Mean_Accuracy", color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Wine Dataset: Classifier Mean Accuracy Across 3 Seeds")
plt.ylabel("Mean Accuracy")
plt.ylim(0.8, 1.02)
plt.tight_layout()
plt.savefig("classifier_comparison_wine.png", dpi=300)
plt.close()

print("Execution complete! 6 CSV files and figure files saved successfully.")
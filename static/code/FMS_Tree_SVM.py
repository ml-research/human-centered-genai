import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ==== Simulated Dataset ====
np.random.seed(42)
X = np.random.randn(1000, 20)
y = (X[:, 5] > 0.5).astype(int)  # Concept strongly tied to feature 5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ==== MONOSEMANTICITY WITH SVM ====
def monosemmetric_svm(X_train, X_test, y_train, y_test):
    # 1. Feature capacity: best single-feature SVM
    feature_accs = []
    for i in range(X_train.shape[1]):
        clf = SVC(kernel="linear")
        clf.fit(X_train[:, [i]], y_train)
        y_pred = clf.predict(X_test[:, [i]])
        feature_accs.append(accuracy_score(y_test, y_pred))
    best_idx = np.argmax(feature_accs)
    accs_0 = feature_accs[best_idx]

    # 2. Local disentanglement
    X_train_local = np.delete(X_train, best_idx, axis=1)
    X_test_local = np.delete(X_test, best_idx, axis=1)
    clf_local = SVC(kernel="linear")
    clf_local.fit(X_train_local, y_train)
    accs_p = accuracy_score(y_test, clf_local.predict(X_test_local))
    mono_local = 2 * (accs_0 - accs_p)
    mono_local = np.clip(mono_local, 0, 1)

    # 3. Global disentanglement
    lr = LogisticRegression(penalty="l2", solver="liblinear")
    lr.fit(X_train, y_train)
    ranked = np.argsort(np.abs(lr.coef_[0]))[::-1]

    accs_cum = []
    for k in range(1, X.shape[1] + 1):
        top_k = ranked[:k]
        clf = SVC(kernel="linear")
        clf.fit(X_train[:, top_k], y_train)
        accs_cum.append(accuracy_score(y_test, clf.predict(X_test[:, top_k])))

    A_n = sum(acc - accs_0 for acc in accs_cum)
    mono_global = 1 - A_n / len(accs_cum)
    mono_global = np.clip(mono_global, 0, 1)

    # 4. Final score
    mono_score = accs_0 * (mono_local + mono_global) / 2
    return {
        "accs_0": accs_0,
        "local": mono_local,
        "global": mono_global,
        "monosemmetric": mono_score,
    }


# ==== MONOSEMANTICITY WITH TREE ====
def monosemmetric_tree(X_train, X_test, y_train, y_test):
    # 1. Feature capacity from root node
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X_train, y_train)
    root_feature = tree.tree_.feature[0]
    accs_0 = accuracy_score(y_test, tree.predict(X_test))

    # 2. Local disentanglement
    X_train_local = np.delete(X_train, root_feature, axis=1)
    X_test_local = np.delete(X_test, root_feature, axis=1)
    tree_local = DecisionTreeClassifier(max_depth=1)
    tree_local.fit(X_train_local, y_train)
    accs_p = accuracy_score(y_test, tree_local.predict(X_test_local))
    mono_local = 2 * (accs_0 - accs_p)
    mono_local = np.clip(mono_local, 0, 1)

    # 3. Global disentanglement: increasing depth
    accs_cum = []
    for d in range(1, X.shape[1] + 1):
        tree = DecisionTreeClassifier(max_depth=d)
        tree.fit(X_train, y_train)
        accs_cum.append(accuracy_score(y_test, tree.predict(X_test)))
        if accs_cum[-1] >= 1 - 1e-3:  # early stopping
            break

    A_n = sum(acc - accs_0 for acc in accs_cum)
    mono_global = 1 - A_n / len(accs_cum)
    mono_global = np.clip(mono_global, 0, 1)

    # 4. Final score
    mono_score = accs_0 * (mono_local + mono_global) / 2
    return {
        "accs_0": accs_0,
        "local": mono_local,
        "global": mono_global,
        "monosemmetric": mono_score,
    }


# ==== Run Both Methods ====
svm_results = monosemmetric_svm(X_train, X_test, y_train, y_test)
tree_results = monosemmetric_tree(X_train, X_test, y_train, y_test)


# ==== Print Results ====
def print_results(name, results):
    print(f"\n{name} Results")
    print("-" * 30)
    print(f"Feature Capacity (accs_0):      {results['accs_0']:.3f}")
    print(f"Local Disentanglement:          {results['local']:.3f}")
    print(f"Global Disentanglement:         {results['global']:.3f}")
    print(f"Final Monosemmetric Score:      {results['monosemmetric']:.3f}")


print_results("SVM-Based", svm_results)
print_results("Tree-Based", tree_results)

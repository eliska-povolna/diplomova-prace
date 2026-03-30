#!/usr/bin/env python3
import pickle
import numpy as np
from scipy.sparse import csr_matrix

print("=" * 80)
print("OVĚŘENÍ FORMÁTU GENEROVANÝCH SOUBORŮ")
print("=" * 80)

# Načti train data
with open("data/processed_yelp_easystudy/processed_train.pkl", "rb") as f:
    X_train = pickle.load(f)
print(f"\n✓ processed_train.pkl:")
print(f"  - Typ: {type(X_train).__name__}")
print(f"  - Tvar: {X_train.shape} (uživatelé × itemy)")
print(f"  - NNZ: {X_train.nnz:,}")
print(f"  - Datový typ: {X_train.dtype}")
assert isinstance(X_train, csr_matrix), "Musí být CSR matrix!"
assert X_train.shape == (443, 9039), "Špatné dimenze!"

# Načti test data
with open("data/processed_yelp_easystudy/processed_test.pkl", "rb") as f:
    X_test = pickle.load(f)
print(f"\n✓ processed_test.pkl:")
print(f"  - Typ: {type(X_test).__name__}")
print(f"  - Tvar: {X_test.shape} (uživatelé × itemy)")
print(f"  - NNZ: {X_test.nnz:,}")
print(f"  - Datový typ: {X_test.dtype}")
assert isinstance(X_test, csr_matrix), "Musí být CSR matrix!"
assert X_test.shape == (50, 9039), "Špatné dimenze!"

# Načti item2index mapping
with open("data/processed_yelp_easystudy/item2index.pkl", "rb") as f:
    item2index = pickle.load(f)
print(f"\n✓ item2index.pkl:")
print(f"  - Typ: {type(item2index).__name__}")
print(f"  - Počet items: {len(item2index):,}")
print(f"  - Příklad mapování: {list(item2index.items())[:3]}")
assert isinstance(item2index, dict), "Musí být slovník!"
assert len(item2index) == 9039, "Špatný počet items!"

# Načti user2index mapping
with open("data/processed_yelp_easystudy/user2index.pkl", "rb") as f:
    user2index = pickle.load(f)
print(f"\n✓ user2index.pkl:")
print(f"  - Typ: {type(user2index).__name__}")
print(f"  - Počet uživatelů: {len(user2index):,}")
print(f"  - Příklad mapování: {list(user2index.items())[:3]}")
assert isinstance(user2index, dict), "Musí být slovník!"

print("\n" + "=" * 80)
print("STATISTIKY DAT")
print("=" * 80)
print(f"Train uživatelé: {X_train.shape[0]}")
print(f"Test uživatelé:  {X_test.shape[0]}")
print(f"Items (businesses): {X_train.shape[1]}")
print(f"Train interakcí: {X_train.nnz:,}")
print(f"Test interakcí:  {X_test.nnz:,}")
print(f"Průměrný počet interakcí na uživatele (train): {X_train.nnz / X_train.shape[0]:.1f}")
print(f"Průměrný počet interakcí na uživatele (test):  {X_test.nnz / X_test.shape[0]:.1f}")
print(f"Průměrný počet interakcí na business: {(X_train.nnz + X_test.nnz) / X_train.shape[1]:.1f}")

print("\n" + "=" * 80)
print("✅ VŠECHNY KONTROLY USPĚLY - DATASET JE PŘIPRAVEN!")
print("=" * 80)

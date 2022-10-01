#!/usr/bin/env python
# Created by "Thieu" at 08:08, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def generate_data(test_size=0.3):
    # Load the data set; In this example, the breast cancer dataset is loaded.
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return {"X_train": X_train_std, "y_train": y_train, "X_test": X_test_std, "y_test": y_test, "scaler": sc}

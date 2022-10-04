#!/usr/bin/env python
# Created by "Thieu" at 05:52, 13/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.preprocessing import LabelEncoder
from src.classification_svc import ClassificationSVC
from src import data_util
from mealpy.swarm_based import WOA
from permetrics.classification import ClassificationMetric


if __name__ == "__main__":
    list_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_encoder = LabelEncoder()
    kernel_encoder.fit(list_kernels)

    data = data_util.generate_data(test_size=0.25)
    data["KERNEL_ENCODER"] = kernel_encoder

    # x1. C: float [0.1 to 10000.0]
    # x2. Kernel: [‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’]

    LB = [0.1, 0.]
    UB = [10000.0, 3.99]

    problem = ClassificationSVC(lb=LB, ub=UB, minmax="max", data=data, save_population=False, log_to="console")

    algorithm = WOA.OriginalWOA(epoch=10, pop_size=20)
    best_position, best_fitness = algorithm.solve(problem)

    best_solution = problem.decode_solution(best_position)

    print(f"Best fitness (accuracy score) value: {best_fitness}")
    print(f"Best parameters: {best_solution}")

    ###### Get the best tuned neural network to predict test set
    best_network = problem.generate_trained_model(best_solution)
    y_pred = best_network.predict(data["X_test"])

    evaluator = ClassificationMetric(data["y_test"], y_pred, decimal=6)
    print(evaluator.get_metrics_by_list_names(["AS", "RS", "PS", "F1S", "F2S"]))

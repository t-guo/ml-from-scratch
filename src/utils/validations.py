def is_fitted(estimator, attribute):
    if not hasattr(estimator, attribute):
        raise Exception('Estimator is not fitted.')


def is_classification(y):
    if not all(isinstance(label, int) for label in y):
        raise Exception('Class labels are not discrete integers.')
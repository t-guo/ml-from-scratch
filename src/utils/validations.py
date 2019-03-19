def is_fitted(estimator, attribute):
    if not hasattr(estimator, attribute):
        raise Exception('Estimator is not fitted.')
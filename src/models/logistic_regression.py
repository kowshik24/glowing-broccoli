from sklearn.linear_model import LogisticRegression

from .registry import register


@register("logistic_regression")
def build(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced", random_state=random_state, max_iter=1000
    )

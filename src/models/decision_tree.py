from sklearn.tree import DecisionTreeClassifier

from .registry import register


@register("decision_tree")
def build(random_state: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(class_weight="balanced", random_state=random_state)

from src.utils import classes


def get_metric_object_by_name(name):
    if name == "bleu":
        return classes.BleuScoreCustomMetric()

    raise ValueError("Chỉ mới định nghĩa cho bleu")

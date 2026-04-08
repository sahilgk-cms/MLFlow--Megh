from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold


class CVFactory:

    @staticmethod
    def create(cv_type: str, **kwargs):
        if cv_type == "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits = kwargs.get("n_splits", 5))
        elif cv_type == "KFold":
            return KFold(n_splits = kwargs.get("n_splits", 5))
        elif cv_type == "StratifiedKFold":
            return StratifiedKFold(n_splits = kwargs.get("n_splits", 5))
        else:
            raise ValueError("CV type not supported")




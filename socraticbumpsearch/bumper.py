from sklearn.utils.validation import check_X_y, check_array
from sklearn.base import RegressorMixin, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, resample
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from joblib import Parallel, delayed


def subtrain(estimator,X, y,random_state,scorer_):
    estimator = clone(estimator)
    X_resample, y_resample = resample(X, y, random_state=random_state)
    estimator.fit(X_resample, y_resample)
    score = scorer_(estimator,X, y)
    return {'estimator':estimator,'score':score}


class BumperBase(MetaEstimatorMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)

        random_state = check_random_state(self.random_state)

        scorer_ = self.get_scorer()

        models_score_list = Parallel(n_jobs=self.n_jobs)(delayed(subtrain)(self.estimator,X, y ,random_state,scorer_) for i in range(self.n_bumps))

        best_model =  models_score_list[0]['estimator']
        best_score =  models_score_list[0]['score']

        for model_score_dic in  models_score_list[1:]:
            if model_score_dic['score'] > best_score:
                best_model = model_score_dic['estimator']
                best_score = model_score_dic['score']

        self.best_estimator_ = best_model
        self.all_models_score_ = models_score_list
        self.is_fitted = True
        
        return self

    def predict(self, X):
        self.check_is_fitted()
        X = check_array(X)
        return self.best_estimator_.predict(X)

    def check_is_fitted(self):
        if self.is_fitted != True:
            raise NotFittedError("""
            Estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator
            """)
            



class BumperClassifier(ClassifierMixin,BumperBase):
    def __init__(self,
                 estimator=DecisionTreeClassifier(),
                 n_bumps=100,
                 random_state=None,
                 n_jobs=1,
                 scorer = None
                 ):
        self.estimator = estimator
        self.n_bumps = n_bumps
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scorer = scorer

        self.is_fitted = False

    def get_scorer(self):
        if self.scorer == None:
            scorer_ = make_scorer(accuracy_score)
        else:
            scorer_ = self.scorer
        return scorer_



class BumperRegressor(RegressorMixin, BumperBase):
    def __init__(self,
                 estimator=DecisionTreeRegressor(),
                 n_bumps=1000,
                 random_state=None,
                 n_jobs=1,
                 scorer = None
                 ):
        self.estimator = estimator
        self.n_bumps = n_bumps
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scorer = scorer

        self.is_fitted = False

    def get_scorer(self):
        if self.scorer == None:
            scorer_ = make_scorer(r2_score)
        else:
            scorer_ = self.scorer
        return scorer_

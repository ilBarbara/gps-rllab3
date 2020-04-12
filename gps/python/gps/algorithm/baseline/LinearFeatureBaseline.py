import numpy as np

class LinearFeatureBaseline(object):
    def __init__(self, reg_coeff=1e-5):
        self._coef = None
        self._reg_coeff = reg_coeff

    def _features(self, state):
        #path: T*dO
        T = state.shape[0]
        cliped = np.clip(state, -10, 10)
        at = np.arange(T).reshape(-1,1) / 100.
        feature = np.concatenate(\
            [cliped, cliped**2, at, at**2,at**3,np.ones((T, 1))], axis=1)
        return feature
    
    def predict(self, state):
        #predict the baseline
        if self._coef is None:
            return np.zeros(state.shape[0])
        return self._features(state).dot(self._coef)

    def fit(self, states, returns):
        #fit the baseline by min least square
        #states: N*T*dO
        #returns: N*T
        featmat = np.concatenate([self._features(state) for state in states])
        returns = np.concatenate([r for r in returns]) 
        self._coef = np.linalg.lstsq(
            featmat.T.dot(featmat) + self._reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(returns)
        )[0]       

import gpytorch    

class CustomPredictiveLogLikelihood(gpytorch.mlls.PredictiveLogLikelihood):
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
            return self.likelihood.log_marginal(target, approximate_dist_f, **kwargs)

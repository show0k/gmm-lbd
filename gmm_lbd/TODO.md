


## GMM Calinon
* Mu:   D x K array representing the centers of the K GMM components.
* Sigma:    D x D x K array representing the covariance matrices of the K GMM components.*
* Priors: 1 x K array representing the prior probabilities of the K GMM components


* Data:    D x N array representing N datapoints of D dimensions.
* prob:  1 x N array representing the probabilities for the N datapoints.     

## GMM scikit
* means_ array, shape (n_components, n_features)
    Mean parameters for each mixture component.

* covars_ Covariance parameters for each mixture component. The shape depends on covariance_type:
    (n_components, n_features)             if 'spherical',
    (n_features, n_features)               if 'tied',
    (n_components, n_features)             if 'diag',
    (n_components, n_features, n_features) if 'full'

* weights_ : array, shape (n_components,)
    This attribute stores the mixing weights for each mixture component.
    



### socraticbumpsearch
A scikit-learn compatible implementation of Bumping as described by “Elements of Statistical Learning” second edition (290-292). 
As consistent with the rest of scikit-learn, I have used joblib to parallelize the search.  

>“We draw bootstrap samples and fit a model to each. But rather than average the predictions, we choose the model estimated from a bootstrap sample that best fits the training data. “(Elements of Statistical Learning” second edition, 291)


Bumming seems to be recommended for noisy data sets.
>“By perturbing the data, bumping tries to move the fitting procedure around to good areas of model space. For example, if a few data points are causing the procedure to find a poor solution, any bootstrap” (Elements of Statistical Learning” second edition, 291).

Install
-------
>pip install git+https://github.com/pr38/socraticbumpsearch

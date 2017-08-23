from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn import svm


class SecPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, pca_n=10, svm_k='rbf'):
        self.pca_n = pca_n
        self.svm_k = svm_k

    # BaseEstimator gives it a .get_params() and .set_params() method that provide the ability to
    # read and override the parameters of the predictor.

    # ClassifierMixin gives it a .score(inputs, expected_result) method that calls .predict(input)
    # and compares it with expected_result

    def fit(self, inputs, targets):
        # do z-transformation, store the mean and std
        print('Begin fitting the model...')
        self.z_mean = inputs.mean(axis=0)
        self.z_std = inputs.std(axis=0)
        inputs_z = (inputs - self.z_mean) / self.z_std

        # do PCA, store the PCA object
        self.pca = PCA(n_components=self.pca_n)
        pca_df = self.pca.fit_transform(inputs_z)

        # fit SVC on PCA-data, store the SVC
        self.classifier = svm.SVC(kernel=self.svm_k)
        self.classifier.fit(pca_df, targets)
        print('Fitting ends.\n')

    def predict(self, inputs):
        print('Begin predicting structures...')
        inputs_z = (inputs - self.z_mean) / self.z_std
        # use the PCA object to transform the input
        pca_df = self.pca.transform(inputs_z)
        print('Prediction ends.\n')
        return self.classifier.predict(pca_df)

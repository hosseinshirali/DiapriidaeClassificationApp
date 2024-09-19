import tensorflow as tf
from keras.models import Model
import numpy as np
import joblib
from pathlib import Path
from keras_cv_attention_models import beit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Queue


BASE_PATH = Path(__file__).parents[1] / 'weights'
SCALER_PATH = BASE_PATH / 'standardscaler.joblib'
OC_SVM_PATH = BASE_PATH / 'ocsvm_model.joblib'
PCA_PATH = BASE_PATH / 'pca.joblib'
AUTOTUNE = tf.data.experimental.AUTOTUNE

np.random.seed(0)
tf.random.set_seed(0)

# hyperparameter; lower (negative) threshold results in
# more inlier and less outlier predictions
# can be seen as the distance to the hyper sphere
# of the One-Class SVM in which samples still
# count as inliers
THRESHOLD = -0.9

def detect_outliers(imgs: np.ndarray, queue: Queue):

    model = beit.BeitV2LargePatch16(
        input_shape=(224,224,3),
        num_classes=0,
        pretrained='imagenet21k-ft1k'
    )

    preprocess_func = model.preprocess_input

    model = Model(model.inputs, model.get_layer('out_ln').output)

    imgs = preprocess_func(imgs)

    imgs = tf.data.Dataset.from_tensor_slices(imgs).batch(4).prefetch(buffer_size=AUTOTUNE)

    features = []

    # feature extraction
    for img_batch in imgs:
        features.extend(model(img_batch))

    features = np.array(features)

    del model

    # Take PCA to reduce feature space dimensionality
    pca: PCA = joblib.load(PCA_PATH)
    features = pca.transform(features)

    # Apply standard scaler to output from beitv2
    ss: StandardScaler = joblib.load(SCALER_PATH)
    features = ss.transform(features)

    oc_svm_clf: svm.OneClassSVM = joblib.load(OC_SVM_PATH)

    oc_svm_preds = oc_svm_clf.decision_function(features)
    is_outlier = [True if score < THRESHOLD else False for score in oc_svm_preds]

    queue.put(is_outlier)
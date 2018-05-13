from __future__ import division
from itertools import repeat
import math
import numpy as np

from sklearn.base import clone, BaseEstimator

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from skimage.feature import blob_doh
from skimage.transform import resize, integral_image
from skimage.feature import haar_like_feature_coord, haar_like_feature

from mahotas.features import zernike_moments, surf


class BlobExtractor(BaseEstimator):
    """Feature extractor using a blob detector.

    This extractor will detect candidate regions using a blob detector,
    i.e. maximum of the determinant of Hessian, and will extract the Zernike's
    moments, the SURF descriptors and Haal-like features.

    Parameters
    ----------
    min_radius : int, default=5
        The minimum radius of the candidate to be detected.

    max_radius : int, default=28
        The maximum radius of the candidate to be detected.

    padding : float, default=2.0
        The region around the blob will be enlarged by the factor given in
        padding.

    iou_threshold : float, default=0.5
        A value between 0 and 1. If the IOU between the candidate and the
        target is greater than this threshold, the candidate is considered as a
        crater. It increases the precision, but decreases the recall.

    blob_threshold : float, default=0.01
        The threshold used to extract the candidate region in the DoH map.
        Values above this threshold will be considered as a ROI.

    overlap : float, default=0.2
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than threshold, the smaller blob is eliminated.

    resized : int, default=20
        The width and height of the resized candidate used to extract haar-like
        features.

    threshold_haar : float, default=0.5
        The proportion of variance explained by the features we don't keep. It
        increases the computation time and the memory needed, but improves
        the performances.

    """

    def __init__(self, min_radius=5, max_radius=28, padding=2.0,
                 iou_threshold=0.5,
                 blob_threshold=.01, overlap=.2, resized=20,
                 threshold_haar=0.5):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.padding = padding
        self.iou_threshold = iou_threshold
        self.blob_threshold = blob_threshold
        self.overlap = overlap
        self.resized = resized
        self.threshold_haar = threshold_haar
        self.feature_coord = None
        self.feature_type = None

    def fit_extract(self, X, y):
        if y is None:
            return self.extract(X, y)
        else:
            return self.fit(X, y).extract(X, y)

    def fit(self, X, y):
        # we fit the haar-like features by choosing the optimal features to keep
        sample_patch, score_iou = self.build_sample_patches(X, y, n=600)
        score_iou[score_iou > self.iou_threshold] = 1
        sample_label = score_iou.astype(np.int)
        select_feat_coord, select_feat_type = self.find_best_haar_features(
            sample_patch, sample_label)
        self.feature_coord = select_feat_coord
        self.feature_type = select_feat_type
        return self

    def extract(self, X, y):
        if y is None:
            y = [y] * X.shape[0]

        features = []
        candidates = []
        labels = []
        idx_image = []
        for idx, (image, craters) in enumerate(zip(X, y)):

            # find candidates
            blobs = blob_doh(image, min_sigma=self.min_radius,
                             max_sigma=self.max_radius,
                             threshold=self.blob_threshold,
                             overlap=self.overlap)

            # convert the candidate to list of tuple
            candidate_blobs = [tuple(blob) for blob in blobs]

            # branch used during testing
            if craters is None:
                labels += [None] * len(candidate_blobs)
            # branch used if there is no crater in the image
            elif len(
                    craters) == 0:
                labels += [0] * len(candidate_blobs)
            else:
                # branch used if we did not detect any blob
                if len(candidate_blobs) == 0:
                    continue
                else:
                    # find the maximum scores between each candidate and
                    # the ground-truth
                    scores_candidates = [
                        max(map(cc_iou, repeat(blob, len(craters)), craters))
                        for blob in candidate_blobs]
                    # threshold the scores
                    labels += [0 if score < self.iou_threshold else 1
                               for score in scores_candidates]
            candidates += candidate_blobs
            features += [self._extract_features(image, blob) for blob in
                         candidate_blobs]
            idx_image += [idx] * len(candidate_blobs)

        return idx_image, features, candidates, labels

    def build_sample_patches(self, X, y, n=600):
        sample = np.random.choice([i for i in range(len(X))],
                                  min(n, X.shape[0]))
        sample_X = X[sample]
        sample_y = y[sample]

        # build a dataset of candidate patches
        l_patch = []
        score_iou = np.array([])
        for image, craters in zip(sample_X, sample_y):

            # get candidate blobs
            blobs = blob_doh(image, min_sigma=self.min_radius,
                             max_sigma=self.max_radius,
                             threshold=self.blob_threshold,
                             overlap=self.overlap)
            candidate_blobs = [tuple(blob) for blob in blobs]

            if len(candidate_blobs) > 0:
                for blob in candidate_blobs:
                    # get patch
                    row, col, radius = int(blob[0]), int(blob[1]), int(blob[2])
                    padded_radius = int(self.padding * radius)
                    col_min = max(col - padded_radius, 0)
                    row_min = max(row - padded_radius, 0)
                    col_max = min(col + padded_radius, image.shape[0] - 1)
                    row_max = min(row + padded_radius, image.shape[1] - 1)
                    patch = image[row_min:row_max, col_min:col_max]
                    resized_patch = resize(patch, (self.resized, self.resized))
                    l_patch.append(resized_patch)

                if len(craters) == 0:
                    scores_candidates = [0] * len(candidate_blobs)
                else:
                    scores_candidates = [
                        max(map(cc_iou, repeat(blob, len(craters)), craters))
                        for blob in candidate_blobs]
                scores_candidates = np.array(scores_candidates)
                score_iou = np.hstack((score_iou, scores_candidates))

        return np.array(l_patch), score_iou

    def find_best_haar_features(self, sample_patches, labels):
        feature_types = ['type-2-x', 'type-2-y']
        feature_coord, feature_type = haar_like_feature_coord(self.resized,
                                                              self.resized,
                                                              feature_types)

        data = []
        for i in range(sample_patches.shape[0]):
            features = extract_feature_image(sample_patches[i], feature_types,
                                             feature_coord=None)
            data.append(features)
        data = np.array(data)

        max_feat = min(100, data.shape[1])
        clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                     max_features=max_feat, n_jobs=-1,
                                     random_state=13)
        clf.fit(data, labels)
        idx_sorted = np.argsort(clf.feature_importances_)[::-1]

        cdf_feature_importances = np.cumsum(
            clf.feature_importances_[idx_sorted[::-1]])
        cdf_feature_importances /= np.max(cdf_feature_importances)
        significant_feature = np.count_nonzero(
            cdf_feature_importances > self.threshold_haar)
        selected_feature_coord = feature_coord[idx_sorted[:significant_feature]]
        selected_feature_type = feature_type[idx_sorted[:significant_feature]]

        return selected_feature_coord, selected_feature_type

    def _extract_features(self, X, candidate):

        row, col, radius = int(candidate[0]), int(candidate[1]), int(
            candidate[2])
        padded_radius = int(self.padding * radius)

        # compute the coordinate of the patch to select
        col_min = max(col - padded_radius, 0)
        row_min = max(row - padded_radius, 0)
        col_max = min(col + padded_radius, X.shape[0] - 1)
        row_max = min(row + padded_radius, X.shape[1] - 1)

        # extract patch
        patch = X[row_min:row_max, col_min:col_max]
        resized_patch = resize(patch, (self.resized, self.resized))

        # compute Zernike moments
        zernike = zernike_moments(patch, radius)

        # compute surf descriptors
        scale_surf = 2 * self.padding * radius / 20
        keypoint = np.array([[row, col, scale_surf, 0.1, 1]])
        surf_descriptor = surf.descriptors(X, keypoint,
                                           is_integral=False).ravel()
        if not surf_descriptor.size:
            surf_descriptor = np.zeros((70,))

        # compute haar-like features
        haar_features = extract_feature_image(resized_patch, self.feature_type,
                                              self.feature_coord)

        return np.hstack((zernike, surf_descriptor, haar_features))


class ObjectDetector(object):
    """Object detector.

    Object detector using an extractor (which is used to extract feature) and
    an estimator.

    Parameters
    ----------
    extractor : object, default=BlobDetector()
        The feature extractor used before to train the estimator.

    estimator : object, default=GradientBoostingClassifier()
        The estimator used to decide if a candidate is a crater or not.

    Attributes
    ----------
    extractor_ : object,
        The actual extractor used after fit.

    estimator_ : object,
        The actual estimator used after fit.

    """

    def __init__(self, extractor=None, estimator=None):
        self.extractor = extractor
        self.estimator = estimator

    def _extract_features(self, X, y):
        if y is None:
            idx_image, features, candidates, _ = self.extractor_.fit_extract(
                X, None)
            idx_image = np.array(idx_image)
            features = np.array(features)
            return features, candidates, idx_image
        else:
            (idx_image, features, candidates,
             labels) = self.extractor_.fit_extract(X, y)
            labels = np.array(labels)
            idx_image = np.array(idx_image)
            features = np.array(features)
            return features, candidates, labels, idx_image

    def fit(self, X, y):
        if self.extractor is None:
            self.extractor_ = BlobExtractor()
        else:
            self.extractor_ = clone(self.extractor)

        if self.estimator is None:
            #self.estimator_ = GradientBoostingClassifier(n_estimators=100)
            max_feat = min(100, X.shape[1])
            self.estimator_ = RandomForestClassifier(n_estimators=1000,
                                                     max_depth=None,
                                                     max_features=max_feat,
                                                     n_jobs=-1,
                                                     random_state=13)
        else:
            self.estimator_ = clone(self.estimator)

        # data augmentation
        # TODO

        # extract the features for the training data
        print("extracting features...")
        features, _, labels, _ = self._extract_features(X, y)

        # fit the underlying classifier
        print("training model")
        self.estimator_.fit(features, labels)

        return self

    def predict(self, X):
        # extract the data for the current image
        features, candidates, idx_image = self._extract_features(X, None)

        # classify each candidate
        y_pred = self.estimator_.predict_proba(features)[:, 1]

        # organize the output
        output = [[] for _ in range(len(X))]
        for predicted_crater, probability, idx in zip(candidates, y_pred,
                                                      idx_image):
            output[idx].append(
                (probability, predicted_crater[0], predicted_crater[1],
                 predicted_crater[2]))

        return np.array(output, dtype=object)


def cc_iou(circle1, circle2):
    """
    Intersection over Union (IoU) between two circles

    Parameters
    ----------
    circle1 : tuple of floats
        first circle parameters (x_pos, y_pos, radius)
    circle2 : tuple of floats
        second circle parameters (x_pos, y_pos, radius)

    Returns
    -------
    float
        ratio between area of intersection and area of union

    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    d = math.hypot(x2 - x1, y2 - y1)

    area_intersection = cc_intersection(d, r1, r2)
    area_union = math.pi * (r1 * r1 + r2 * r2) - area_intersection

    return area_intersection / area_union


def cc_intersection(dist, rad1, rad2):
    """
    Area of intersection between two circles

    Parameters
    ----------
    dist : positive float
        distance between circle centers
    rad1 : positive float
        radius of first circle
    rad2 : positive float
        radius of second circle

    Returns
    -------
    intersection_area : positive float
        area of intersection between circles

    References
    ----------
    http://mathworld.wolfram.com/Circle-CircleIntersection.html

    """
    if dist < 0:
        raise ValueError("Distance between circles must be positive")
    if rad1 < 0 or rad2 < 0:
        raise ValueError("Circle radius must be positive")

    if dist == 0 or (dist <= abs(rad2 - rad1)):
        return min(rad1, rad2) ** 2 * math.pi

    if dist > rad1 + rad2 or rad1 == 0 or rad2 == 0:
        return 0

    rad1_sq = rad1 * rad1
    rad2_sq = rad2 * rad2

    circle1 = rad1_sq * math.acos((dist * dist + rad1_sq - rad2_sq) /
                                  (2 * dist * rad1))
    circle2 = rad2_sq * math.acos((dist * dist + rad2_sq - rad1_sq) /
                                  (2 * dist * rad2))
    intersec = 0.5 * math.sqrt((-dist + rad1 + rad2) * (dist + rad1 - rad2) *
                               (dist - rad1 + rad2) * (dist + rad1 + rad2))
    intersection_area = circle1 + circle2 + intersec

    return intersection_area


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def extract_feature_image(image, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(image)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

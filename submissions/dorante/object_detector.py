from __future__ import division
import math
from itertools import repeat
import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import blob_log, blob_doh
from mahotas.features import zernike_moments
from skimage.feature import canny
from skimage import filters

###############################################################################
# Extractor


class BlobExtractor(BaseEstimator):
    """Feature extractor using a blob detector.

    This extractor will detect candidate regions using a blob detector,
    i.e. maximum of the determinant of Hessian, and will extract the Zernike's
    moments.

    Parameters
    ----------
    min_radius : int, default=5
        The minimum radius of the candidate to be detected.

    max_radius : int, default=30
        The maximum radius of the candidate to be detected.

    padding : float, default=2.0
        The region around the blob will be enlarged by the factor given in
        padding.

    iou_threshold : float, default=0.4
        A value between 0 and 1. If the IOU between the candidate and the
        target is greater than this threshold, the candidate is considered as a
        crater.

    """

    def __init__(self, min_radius=5, max_radius=28, padding=2.0,
                 iou_threshold=0.4, doh=False):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.padding = padding
        self.iou_threshold = iou_threshold
        self.doh = doh

    def fit(self, X, y=None, **fit_params):
        # This extractor does not require any fitting
        return self

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
        patch_edge = filters.sobel(patch)

        # compute Zernike moments
        zernike = zernike_moments(patch, radius)
        zernike_edge = zernike_moments(patch_edge, radius)

        return np.hstack((zernike, zernike_edge))

    def extract(self, X, y=None, **fit_params):
        # find candidates
        if self.doh:
            blobs = blob_doh(X, min_sigma=self.min_radius,
                             max_sigma=self.max_radius, threshold=.1)
        else:
            blobs = blob_log(X, min_sigma=1, max_sigma=30, num_sigma=10,
                             threshold=.1)
            blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
            blobs = blobs[blobs[:, 2] >= self.min_radius]
            blobs = blobs[blobs[:, 2] <= self.max_radius]

        # convert the candidate to list of tuple
        candidate_blobs = [tuple(blob) for blob in blobs]

        # extract feature to be returned
        features = [self._extract_features(X, blob)
                    for blob in candidate_blobs]

        if y is None:
            # branch used during testing
            return features, candidate_blobs, [None] * len(features)
        elif not y:
            # branch if there is no crater in the image
            labels = [0] * len(candidate_blobs)

            return features, candidate_blobs, labels
        else:
            # case the we did not detect any blobs
            if not len(features):
                return ([], [], [])

            # find the maximum scores between each candidate and the
            # ground-truth
            scores_candidates = [max(map(cc_iou, repeat(blob, len(y)), y))
                                 for blob in candidate_blobs]

            # threshold the scores
            labels = [0 if score < self.iou_threshold else 1
                      for score in scores_candidates]

            return features, candidate_blobs, labels

    def fit_extract(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).extract(X)
        else:
            return self.fit(X, y, **fit_params).extract(X, y)


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
        # extract feature for all the image containing craters
        data_extracted = [self.extractor_.fit_extract(image, craters)
                          for image, craters in zip(X, y)]

        # organize the data to fit it inside the classifier
        data, location, target, idx_cand_to_img = [], [], [], []
        for img_idx, candidate in enumerate(data_extracted):
            # check if this is an empty features
            if len(candidate[0]):
                data.append(np.vstack(candidate[0]))
                location += candidate[1]
                target += candidate[2]
                idx_cand_to_img += [img_idx] * len(candidate[1])
        # convert to numpy array the data needed to feed the classifier
        data = np.concatenate(data)
        target = np.array(target)

        return data, location, target, idx_cand_to_img

    def fit(self, X, y):
        if self.extractor is None:
            self.extractor_ = BlobExtractor()
        else:
            self.extractor_ = clone(self.extractor)

        if self.estimator is None:
            self.estimator_ = GradientBoostingClassifier(n_estimators=100)
        else:
            self.estimator_ = clone(self.estimator)

        # extract the features for the training data
        print("extracting features...")
        data, _, target, _ = self._extract_features(X, y)

        # fit the underlying classifier
        print("training model")
        self.estimator_.fit(data, target)

        return self

    def predict(self, X):
        # extract the data for the current image
        data, location, _, idx_cand_to_img = self._extract_features(
            X, [None] * len(X))

        # classify each candidate
        y_pred = self.estimator_.predict_proba(data)

        # organize the output
        output = [[] for _ in range(len(X))]
        crater_idx = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        for crater, pred, img_idx in zip(location, y_pred, idx_cand_to_img):
            output[img_idx].append((pred[crater_idx] / 0.04,
                                    crater[0], crater[1], crater[2]))

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

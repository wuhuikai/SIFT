#ifndef SIFT
#define SIFT

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/*****************************SIFT*********************************************/

/**
 * DEFAULT SIFT PARAMS
 */
#define SIFT_INTERVALS 3
#define SIFT_SIGMA 1.6
#define SIFT_CONTRAST_THRES 0.04
#define SIFT_CURVATURE_THRES 10
#define SIFT_IMG_DBL true
#define SIFT_DESCR_WIDTH 4
#define SIFT_DESCR_HIST_BINS 8

/**
 * [Detect & extract sift features for an image]
 *
 * @param img             [source image for extract sift features : grayscale image with pixel values in 0.0f~1.0f]
 * @param keypoints       [vector for store sift feature point]
 * @param descriptor      [Mat for store sift descriptor, each row for a keypoint]
 * @param intervals       [the number of sampled intervals per octave]
 * @param sigma           [sigma for initial gaussian smoothing]
 * @param contrast_thres  [threshold on keypoint contrast]
 * @param curvature_thres [threshold on keypoint ratio of principle curvatures]
 * @param img_dbl         [double image size before pyramid construction?]
 * @param descr_width     [width of descriptor histogram array]
 * @param descr_hist_bins [number of bins per histogram in descriptor array]
 */
void extractSiftFeatures(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor, int intervals = SIFT_INTERVALS,
                         double sigma = SIFT_SIGMA, double contrast_thres = SIFT_CONTRAST_THRES,
                         int curvature_thres = SIFT_CURVATURE_THRES, bool img_dbl = SIFT_IMG_DBL,
                         int descr_width = SIFT_DESCR_WIDTH, int descr_hist_bins = SIFT_DESCR_HIST_BINS);

#endif
#ifndef SIFT
#define SIFT

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>

using namespace cv;
using namespace std;

struct Feature {
	float __x;
	float __y;
	float __scl;

	int __r;
	int __c;
	int __interval;
	int __octave;
	int __idx;

	float __sub_interval;
	float __scl_octave;

	float __ori;
};
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
 * SIFT PARAMS SETDOWN
 */
#define SIFT_INIT_SIGMA 0.5
#define SIFT_IMG_BORDER 5
#define SIFT_MAX_INTERP_STEPS 5

#define SIFT_ORI_HIST_BINS 36
#define SIFT_ORI_SIG_FCTR 1.5
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR
#define SIFT_ORI_SMOOTH_PASSES 2
#define SIFT_ORI_PEAK_RATIO 0.8

/**
 * [Detect & extract sift features for an image]
 *
 * @param img             [source image for extract sift features : grayscale image with pixel values in 0.0f~1.0f]
 * @param feats           [vector for store sift features extracted]
 * @param intervals       [the number of sampled intervals per octave]
 * @param sigma           [sigma for initial gaussian smoothing]
 * @param contrast_thres  [threshold on keypoint contrast]
 * @param curvature_thres [threshold on keypoint ratio of principle curvatures]
 * @param img_dbl         [double image size before pyramid construction?]
 * @param descr_width     [width of descriptor histogram array]
 * @param descr_hist_bins [number of bins per histogram in descriptor array]
 */
void extractSiftFeatures(const Mat &img, vector<Feature> &feats, int intervals = SIFT_INTERVALS,
                         double sigma = SIFT_SIGMA, double contrast_thres = SIFT_CONTRAST_THRES,
                         int curvature_thres = SIFT_CURVATURE_THRES, bool img_dbl = SIFT_IMG_DBL,
                         int descr_width = SIFT_DESCR_WIDTH, int descr_hist_bins = SIFT_DESCR_HIST_BINS);

/**
 * [Create the initial image for builing gaussian pyramid, optionally double the image size before smoothing]
 *
 * @param img      [input image]
 * @param img_dbl  [double size before smoothing?]
 * @param sigma    [sigma for gaussian smoothing]
 *
 * @return         [image returned]
 */
static Mat __createInitImg(const Mat &img, bool img_dbl, double sigma);

/**
 * [Build Gaussian pyramid]
 *
 * @param base             [base image for the pyramid]
 * @param gaussian_pyramid [returned gaussian pyramid, size : octaves x (intervals + 3)]
 * @param octaves          [number of octaves]
 * @param intervals        [number of intervals per octave]
 * @param sigma            [sigma for gaussian smoothing]
 */
static void __buildGaussPyramid(const Mat &base, vector<Mat> &gaussian_pyramid, int octaves, int intervals, double sigma);

/**
 * [Build Difference of Gaussian Pyramid]
 *
 * @param gaussian_pyramid [gaussian pyramid]
 * @param dog_pyramid      [returned dog pyramid, size : octaves x (intervals + 2)]
 * @param octaves          [number of octaves]
 * @param intervals        [number of intervals per octave]
 */
static void __buildDogPyramid(const vector<Mat> &gaussian_pyramid, vector<Mat> &dog_pyramid, int octaves, int intervals);

/**
 * [Detects features at extrema in DoG scale space]
 *
 * @param dog_pyramid     [Dog pyramid]
 * @param feats           [Detected features with scales, origentations and descriptors to be determined]
 * @param octaves         [number of octaves]
 * @param intervals       [number of intervals per octave]
 * @param contrast_thres  [low threshold on feature contrast]
 * @param curvature_thres [high threshold on feature ratio of principal curvatures]
 */
static void __scaleSpaceExtrema(const vector<Mat> &dog_pyramid, vector<Feature> &feats, int octaves, int intervals,
                                double contrast_thres, int curvature_thres);

/**
 * [Determines whether a pixel is a scale-space extremum]
 *
 * @param  dog_pyramid  [Dog pyramid]
 * @param  idx          [index of intervals]
 * @param  r            [pixel's image row]
 * @param  c            [pixel's image col]
 *
 * @return              [extremum?]
 */
static bool __isExtremum(const vector<Mat> &dog_pyramid, int idx, int r, int c);

/**
 * [Interpolates a scale-space extremum's location]
 *
 * @param  dog_pyramid    [Dog pyramid]
 * @param  feat           [returned feature]
 * @param  idx            [index of intervals]
 * @param  r              [pixel's image row]
 * @param  c              [pixel's image col]
 * @param  intervals      [number of intervals per octave]
 * @param  contrast_thres [low threshold on feature contrast]
 *
 * @return                [is a feature?]
 */
static bool __interpExtremum(const vector<Mat> &dog_pyramid, Feature &feat, int idx, int r, int c, int intervals, double contrast_thres);

/**
 * [one step of extremum interpolation]
 * @param dog_pyramid [Dog pyramid]
 * @param idx         [index of intervals]
 * @param r           [pixel's image row]
 * @param c           [pixel's image col]
 * @param xi          [interpolated subpixel increment to interval]
 * @param xr          [interpolated subpixel increment to row]
 * @param xc          [interpolated subpixel increment to col]
 */
static void __interpStep(const vector<Mat> &dog_pyramid, int idx, int r, int c, double &xi, double &xr, double &xc);

/**
 * [partial derivatives in x, y, interval]
 *
 * @param dog_pyramid [Dog pyramid]
 * @param idx         [index of intervals]
 * @param r           [pixel's image row]
 * @param c           [pixel's image col]
 *
 * @return            [partial derivatives]
 */
static Mat __derivative(const vector<Mat> &dog_pyramid, int idx, int r, int c);

/**
 * [3D Hessian matrix in x, y, interval]
 *
 * @param dog_pyramid [Dog pyramid]
 * @param idx         [index of intervals]
 * @param r           [pixel's image row]
 * @param c           [pixel's image col]
 *
 * @return            [3D Hessian matrix]
 */
static Mat __hessian(const vector<Mat> &dog_pyramid, int idx, int r, int c);

/**
 * [Calculates interpolated pixel contrast]
 *
 * @param dog_pyramid [Dog pyramid]
 * @param idx         [index of intervals]
 * @param r           [pixel's image row]
 * @param c           [pixel's image col]
 * @param xi          [interpolated subpixel increment to interval]
 * @param xr          [interpolated subpixel increment to row]
 * @param xc          [interpolated subpixel increment to col]
 *
 * @return            [interpolated contrast]
 */
static double __interpContrast(const vector<Mat> &dog_pyramid, int idx, int r, int c, double xi, double xr, double xc);

/**
 * [Is a feature too edge like?]
 *
 * @param  dog             [Dog layer]
 * @param  r               [row]
 * @param  c               [col]
 * @param  curvature_thres [high threshold on ratio of principal curvatures]
 *
 * @return                 [edge?]
 */
static bool __isTooEdgeLike(const Mat &dog, int r, int c, int curvature_thres);

/**
 * [Calculates characteristic scale]
 *
 * @param feats     [feature vector]
 * @param sigma     [sigma for Gaussian smoothing]
 * @param intervals [intervals per octave]
 */
static void __calcFeatureScales(vector<Feature> &feats, double sigma, int intervals);

/**
 * [Halves feature coordinates and scale]
 *
 * @param feats [feature vector]
 */
static void __adjustForImgDbl(vector<Feature> &feats);

/**
 * [Computes a canonical orientation for each image feature]
 *
 * @param feats            [feature vector]
 * @param gaussian_pyramid [gaussian pyramid]
 * @param layer_per_octave [number of layers per octave]
 */
static void __calcFeatureOris(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave);

/**
 * [Compute gradient orientation histogram]
 *
 * @param gaussian [gaussian image]
 * @param hist     [returned hist]
 * @param r        [row]
 * @param c        [col]
 * @param n        [number of bins]
 * @param rad      [radius of selected region]
 * @param sigma    [sigma for gaussian smoothing]
 */
static void __oriHist(const Mat &gaussian, vector<double> &hist, int r, int c, int rad, double sigma);

/**
 * [Calculates the gradient magnitude and orientation]
 *
 * @param  gaussian [Gaussian image]
 * @param  r        [row]
 * @param  c        [col]
 * @param  mag      [returned gradient magnitude]
 * @param  ori      [returned gradient orientation]
 *
 * @return          [valid pixel]
 */
static bool __calcGradMagOri(const Mat &gaussian, int r, int c, double &mag, double &ori);

/**
 * [Gaussian smooths an orientation histogram.]
 * @param hist [orientation histogram]
 */
static void __smoothOriHist(vector<double> &hist);

/**
 * [add features with orientation greater than mag_thres]
 *
 * @param feat_queue [queue for features]
 * @param hist       [orientation histogram]
 * @param mag_thres  [magnitude threshold]
 * @param feat       [template feat]
 */
static void __addGoodOriFeatures(queue<Feature> &feat_queue, const vector<double> &hist, double mag_thres, const Feature &feat);

#endif
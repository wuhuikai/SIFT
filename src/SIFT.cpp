#include <iostream>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "SIFT.hpp"

int main() {
	const char* file = "imgs/demo.jpg";

	Mat img = imread(file);
	cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img, CV_32FC1, 1.0 / 255);

	vector<Feature> feats;
	extractSiftFeatures(img, feats);




	/******************************SHOW****************************************/
	img = imread(file);

	// OPENCV
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(img, keypoints);
	Mat output;
	drawKeypoints(img, keypoints, output);
	cout << "Opencv:" << keypoints.size() << endl;
	imshow("OPEN-SIFT", output);

	// MINE
	cout << "Mine:" << feats.size() << endl;
	Scalar colors[] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255)};
	for (auto it = feats.begin(); it != feats.end(); it ++) {
		float x = (*it).__x;
		float y = (*it).__y;
		int octave = (*it).__octave;
		float s = max(8.0 / pow(2.0, octave), 1.0);
		rectangle(img, Point(x - s, y - s), Point(x + s, y + s), colors[octave % 6]);
	}
	imshow("MINE-SIFT", img);
	waitKey();
	imwrite("SIFT.jpg", img);

	return 0;
}

void extractSiftFeatures(const Mat &img, vector<Feature> &feats, int intervals, double sigma,
                         double contrast_thres, int curvature_thres, bool img_dbl, int descr_width, int descr_hist_bins) {
	Mat init_img = __createInitImg(img, img_dbl, sigma);

	// Smallest dimension of top level is ~ 4 pixels
	Size s = init_img.size();
	int octaves = log(min(s.width, s.height)) / log(2) - 2;

	vector<Mat> gaussian_pyramid;
	__buildGaussPyramid(init_img, gaussian_pyramid, octaves, intervals, sigma);

	vector<Mat> dog_pyramid;
	__buildDogPyramid(gaussian_pyramid, dog_pyramid, octaves, intervals);

	__scaleSpaceExtrema(dog_pyramid, feats, octaves, intervals, contrast_thres, curvature_thres);

	__calcFeatureScales(feats, sigma, intervals);

	if (img_dbl)
		__adjustForImgDbl(feats);

	__calcFeatureOris(feats, gaussian_pyramid, intervals + 3);
}

static Mat __createInitImg(const Mat &img, bool img_dbl, double sigma) {
	if (img_dbl) {
		Mat init_img(img.size() * 2, CV_32FC1);
		resize(img, init_img, init_img.size(), 0, 0, INTER_CUBIC);

		double sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
		GaussianBlur(init_img, init_img, Size(), sig_diff, sig_diff);
		return init_img;
	} else {
		Mat init_img = Mat(img.size(), CV_32FC1);

		double sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
		GaussianBlur(img, init_img, Size(), sig_diff, sig_diff);
		return init_img;
	}
}

static void __buildGaussPyramid(const Mat &base, vector<Mat> &gaussian_pyramid, int octaves, int intervals, double sigma) {
	int layer_per_octave = intervals + 3;
	vector<double> sigmas(layer_per_octave);
	double k = pow(2.0f, 1.0f / intervals);

	// increamental sigma
	sigmas[0] = sigma;
	sigmas[1] = sigma * sqrt( k * k - 1 );
	for (int i = 2; i < layer_per_octave; i++)
		sigmas[i] = sigmas[i - 1] * k;

	int layers = octaves * layer_per_octave;
	gaussian_pyramid.reserve(layers);

	for (int oct = 0; oct < octaves; oct ++) {
		for (int lay = 0; lay < layer_per_octave; lay++) {
			if (oct == 0 && lay == 0) {
				gaussian_pyramid.push_back(base.clone());
				continue;
			}

			if (lay == 0) {
				const Mat &last = gaussian_pyramid.back();
				Size s = last.size();
				Mat down_img(s.height / 2, s.width / 2, CV_32FC1);
				resize(last, down_img, down_img.size(), 0, 0, CV_INTER_NN);
				gaussian_pyramid.push_back(down_img);
				continue;
			}

			const Mat &last = gaussian_pyramid.back();
			Mat smooth_img(last.size(), CV_32FC1);
			GaussianBlur(last, smooth_img, Size(), sigmas[lay], sigmas[lay]);
			gaussian_pyramid.push_back(smooth_img);
		}
	}
}

static void __buildDogPyramid(const vector<Mat> &gaussian_pyramid, vector<Mat> &dog_pyramid, int octaves, int intervals) {
	int layer_per_octave_dog = intervals + 2;
	int layer_per_octave_gaussian = intervals + 3;
	dog_pyramid.reserve(octaves * layer_per_octave_dog);

	for (int oct = 0; oct < octaves; oct ++) {
		for (int lay = 0; lay < layer_per_octave_dog; lay ++) {
			int idx = oct * layer_per_octave_gaussian + lay;
			dog_pyramid.push_back(gaussian_pyramid[idx + 1] - gaussian_pyramid[idx]);
		}
	}
}

static void __scaleSpaceExtrema(const vector<Mat> &dog_pyramid, vector<Feature> &feats, int octaves, int intervals,
                                double contrast_thres, int curvature_thres) {
	double prelim_contrast_thres = 0.5 * contrast_thres / intervals;

	int layer_per_octave_dog = intervals + 2;
	for (int oct = 0; oct < octaves; oct ++) {
		Size s = dog_pyramid[oct * layer_per_octave_dog].size();
		for (int lay = 1; lay <= intervals; lay ++) {
			int idx = oct * layer_per_octave_dog + lay;
			const Mat &dog = dog_pyramid[idx];

			for (int r = SIFT_IMG_BORDER; r < s.height - SIFT_IMG_BORDER; r++) {
				for (int c = SIFT_IMG_BORDER; c < s.width - SIFT_IMG_BORDER; c++) {
					float pixel_val = dog.at<float>(r, c);
					if (abs(pixel_val) <= prelim_contrast_thres)
						continue;
					if (!__isExtremum(dog_pyramid, idx, r, c))
						continue;

					Feature feat;
					if (!__interpExtremum(dog_pyramid, feat, idx, r, c, intervals, contrast_thres))
						continue;
					if (__isTooEdgeLike(dog_pyramid[feat.__idx], feat.__r, feat.__c, curvature_thres))
						continue;

					feats.push_back(feat);
				}
			}
		}
	}
}

static bool __isExtremum(const vector<Mat> &dog_pyramid, int idx, int r, int c) {
	float pixel_val = dog_pyramid[idx].at<float>(r, c);

	if (pixel_val > 0) {
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j ++) {
				for (int k = -1; k <= 1; k ++) {
					if (pixel_val < dog_pyramid[idx + i].at<float>(r + j, c + k)) {
						return false;
					}
				}
			}
		}
	} else {
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j ++) {
				for (int k = -1; k <= 1; k ++) {
					if (pixel_val > dog_pyramid[idx + i].at<float>(r + j, c + k)) {
						return false;
					}
				}
			}
		}
	}

	return true;
}

static bool __interpExtremum(const vector<Mat> &dog_pyramid, Feature &feat, int idx, int r, int c, int intervals, double contrast_thres) {
	int layer_per_octave_dog = intervals + 2;
	Size s = dog_pyramid[idx].size();

	int i = 0;
	double xi, xr, xc;
	while ( i < SIFT_MAX_INTERP_STEPS ) {
		__interpStep(dog_pyramid, idx, r, c, xi, xr, xc);
		if (abs(xi) < 0.5  &&  abs(xr) < 0.5  &&  abs(xc) < 0.5)
			break;

		int lay = idx % layer_per_octave_dog + round(xi);
		c += round( xc );
		r += round( xr );
		idx += round( xi );

		if (lay < 1  || lay > intervals || c < SIFT_IMG_BORDER || r < SIFT_IMG_BORDER ||
		        c >= s.width - SIFT_IMG_BORDER || r >= s.height - SIFT_IMG_BORDER) {
			return false;
		}

		i ++;
	}

	if ( i >= SIFT_MAX_INTERP_STEPS )
		return false;

	double contrast = __interpContrast(dog_pyramid, idx, r, c, xi, xr, xc);
	if (abs( contrast ) < contrast_thres / intervals)
		return false;

	int octave = idx / layer_per_octave_dog;
	feat.__x = (c + xc) * pow(2.0, octave);
	feat.__y = (r + xr) * pow(2.0, octave);

	feat.__r = r;
	feat.__c = c;
	feat.__octave = octave;
	feat.__interval = idx % layer_per_octave_dog;
	feat.__idx = idx;
	feat.__sub_interval = xi;

	return true;
}

static void __interpStep(const vector<Mat> &dog_pyramid, int idx, int r, int c, double &xi, double &xr, double &xc) {
	Mat dD = __derivative(dog_pyramid, idx, r, c);
	Mat H = __hessian(dog_pyramid, idx, r, c);
	Mat H_inv = H.inv(DECOMP_SVD);
	Mat X = - H_inv * dD;

	xi = X.at<double>(2, 0);
	xr = X.at<double>(1, 0);
	xc = X.at<double>(0, 0);
}

static Mat __derivative(const vector<Mat> &dog_pyramid, int idx, int r, int c) {
	Mat dI(3, 1, CV_64FC1);
	dI.at<double>(0, 0) = (dog_pyramid[idx].at<float>(r, c + 1) - dog_pyramid[idx].at<float>(r, c - 1)) / 2.0;
	dI.at<double>(1, 0) = (dog_pyramid[idx].at<float>(r + 1, c) - dog_pyramid[idx].at<float>(r - 1, c)) / 2.0;
	dI.at<double>(2, 0) = (dog_pyramid[idx + 1].at<float>(r, c) - dog_pyramid[idx - 1].at<float>(r, c)) / 2.0;

	return dI;
}

static Mat __hessian(const vector<Mat> &dog_pyramid, int idx, int r, int c) {
	Mat H(3, 3, CV_64FC1);

	double v = dog_pyramid[idx].at<float>(r, c);
	double dxx = dog_pyramid[idx].at<float>(r, c + 1) + dog_pyramid[idx].at<float>(r, c - 1) - 2 * v;
	double dyy = dog_pyramid[idx].at<float>(r + 1, c) + dog_pyramid[idx].at<float>(r - 1, c) - 2 * v;
	double dss = dog_pyramid[idx + 1].at<float>(r, c) + dog_pyramid[idx - 1].at<float>(r, c) - 2 * v;
	double dxy = (dog_pyramid[idx].at<float>(r + 1, c + 1) - dog_pyramid[idx].at<float>(r + 1, c - 1) -
	              dog_pyramid[idx].at<float>(r - 1, c + 1) + dog_pyramid[idx].at<float>(r - 1, c - 1)) / 4.0;
	double dxs = (dog_pyramid[idx + 1].at<float>(r, c + 1) - dog_pyramid[idx + 1].at<float>(r, c - 1) -
	              dog_pyramid[idx - 1].at<float>(r, c + 1) + dog_pyramid[idx - 1].at<float>(r, c - 1)) / 4.0;
	double dys = (dog_pyramid[idx + 1].at<float>(r + 1, c) - dog_pyramid[idx + 1].at<float>(r - 1, c) -
	              dog_pyramid[idx - 1].at<float>(r + 1, c) + dog_pyramid[idx - 1].at<float>(r - 1, c)) / 4.0;

	H.at<double>(0, 0) = dxx;
	H.at<double>(0, 1) = dxy;
	H.at<double>(0, 2) = dxs;
	H.at<double>(1, 0) = dxy;
	H.at<double>(1, 1) = dyy;
	H.at<double>(1, 2) = dys;
	H.at<double>(2, 0) = dxs;
	H.at<double>(2, 1) = dys;
	H.at<double>(2, 2) = dss;

	return H;
}

static double __interpContrast(const vector<Mat> &dog_pyramid, int idx, int r, int c, double xi, double xr, double xc) {
	Mat dD = __derivative(dog_pyramid, idx, r, c);
	Mat X(3, 1, CV_64FC1);
	X.at<double>(2, 0) = xi;
	X.at<double>(1, 0) = xr;
	X.at<double>(0, 0) = xc;

	Mat t = dD.t() * X;
	return dog_pyramid[idx].at<float>(r, c) + t.at<double>(0, 0) * 0.5;
}

static bool __isTooEdgeLike(const Mat &dog, int r, int c, int curvature_thres) {
	double d = dog.at<float>(r, c);
	double dxx = dog.at<float>(r, c + 1) + dog.at<float>(r, c - 1) - 2 * d;
	double dyy = dog.at<float>(r + 1, c) + dog.at<float>(r - 1, c) - 2 * d;
	double dxy = (dog.at<float>(r + 1, c + 1) - dog.at<float>(r + 1, c - 1) -
	              dog.at<float>(r - 1, c + 1) + dog.at<float>(r - 1, c - 1)) / 4.0;
	double tr = dxx + dyy;
	double det = dxx * dyy - dxy * dxy;

	// negative determinant -> curvatures have different signs; reject feature
	if ( det <= 0 )
		return true;
	if ( tr * tr / det < ( curvature_thres + 1.0 ) * ( curvature_thres + 1.0 ) / curvature_thres )
		return false;
	return true;
}

static void __calcFeatureScales(vector<Feature> &feats, double sigma, int intervals) {
	int layer_per_octave_dog = intervals + 2;

	for (auto it = feats.begin(); it != feats.end(); it ++) {
		float interval = (*it).__interval + (*it).__sub_interval;
		(*it).__scl = sigma * pow(2.0, (*it).__octave + interval / intervals);
		(*it).__scl_octave = sigma * pow(2.0, interval / intervals);
	}
}

static void __adjustForImgDbl(vector<Feature> &feats) {
	for (auto it = feats.begin(); it != feats.end(); it ++) {
		(*it).__x /= 2.0;
		(*it).__y /= 2.0;
		(*it).__scl /= 2.0;
	}
}

static void __calcFeatureOris(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave) {
	queue<Feature> feat_queue;
	size_t n = feats.size();
	for (size_t i = 0; i < n; i ++) {
		feat_queue.push(feats[i]);
	}

	vector<double> hist(SIFT_ORI_HIST_BINS);
	for (size_t i = 0; i < n; i ++) {
		Feature feat = feat_queue.front();
		feat_queue.pop();

		for (int c = 0; c < SIFT_ORI_HIST_BINS; c ++)
			hist[c] = 0;
		__oriHist(gaussian_pyramid[feat.__octave * layer_per_octave + feat.__interval], hist, feat.__r, feat.__c,
		          round(SIFT_ORI_RADIUS * feat.__scl_octave), SIFT_ORI_SIG_FCTR * feat.__scl_octave);

		for (int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
			__smoothOriHist(hist);

		double max_ori = *max_element(hist.begin(), hist.end());

		__addGoodOriFeatures(feat_queue, hist, max_ori * SIFT_ORI_PEAK_RATIO, feat);
	}

	n = feat_queue.size();
	feats = vector<Feature>(n);
	for (size_t i = 0; i < n; i ++) {
		feats[i] = feat_queue.front();
		feat_queue.pop();
	}
}

static void __oriHist(const Mat &gaussian, vector<double> &hist, int r, int c, int rad, double sigma) {
	double PI_2 = CV_PI * 2.0;

	double exp_denom = 2.0 * sigma * sigma;
	double mag, ori;
	int n = hist.size();
	for (int i = -rad; i <= rad; i++) {
		for (int j = -rad; j <= rad; j++) {
			if (__calcGradMagOri(gaussian, r + i, c + j, mag, ori)) {
				double w = exp(-( i * i + j * j ) / exp_denom);
				int bin = round( n * ( ori + CV_PI ) / PI_2 );
				bin = ( bin < n ) ? bin : 0;
				hist[bin] += w * mag;
			}
		}
	}
}

static bool __calcGradMagOri(const Mat &gaussian, int r, int c, double &mag, double &ori) {
	Size s = gaussian.size();

	if (r > 0  &&  r < s.height - 1  &&  c > 0  &&  c < s.width - 1) {
		float dx = gaussian.at<float>(r, c + 1) - gaussian.at<float>(r, c - 1);
		float dy = gaussian.at<float>(r - 1, c) - gaussian.at<float>(r + 1, c);
		mag = sqrt( dx * dx + dy * dy );
		ori = atan2( dy, dx );
		return true;
	}

	return false;
}

static void __smoothOriHist(vector<double> &hist) {
	int n = hist.size();
	for (int i = 0; i < n; i++ ) {
		hist[i] = 0.25 * hist[(i + n - 1) % n] + 0.5 * hist[i] + 0.25 * hist[(i + 1) % n];
	}
}

static void __addGoodOriFeatures(queue<Feature> &feat_queue, const vector<double> &hist, double mag_thres, const Feature &feat) {
	double PI_2 = CV_PI * 2.0;

	int n = hist.size();
	for (int i = 0; i < n; i++) {
		int l = (i == 0) ? n - 1 : i - 1;
		int r = (i + 1) % n;

		if (hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thres) {
			double bin = i + 0.5 * (hist[l] - hist[r]) / (hist[l] - 2.0 * hist[i] + hist[r]);
			bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
			Feature new_feat = feat;
			new_feat.__ori = PI_2 * bin / n - CV_PI;
			feat_queue.push(new_feat);
		}
	}
}
#include <iostream>
#include <algorithm>

#include "SIFT.hpp"
#include "__SIFT.hpp"

void extractSiftFeatures(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor, int intervals, double sigma,
                         double contrast_thres, int curvature_thres, bool img_dbl, int descr_width, int descr_hist_bins) {
	Mat init_img = __createInitImg(img, img_dbl, sigma);

	// Smallest dimension of top level is ~ 4 pixels
	Size s = init_img.size();
	int octaves = log(min(s.width, s.height)) / log(2) - 2;

	vector<Mat> gaussian_pyramid;
	__buildGaussPyramid(init_img, gaussian_pyramid, octaves, intervals, sigma);

	vector<Mat> dog_pyramid;
	__buildDogPyramid(gaussian_pyramid, dog_pyramid, octaves, intervals);


	vector<Feature> feats;
	__scaleSpaceExtrema(dog_pyramid, feats, octaves, intervals, contrast_thres, curvature_thres);

	__calcFeatureScales(feats, sigma, intervals);

	if (img_dbl)
		__adjustForImgDbl(feats);

	__calcFeatureOris(feats, gaussian_pyramid, intervals + 3);

	__computeDescriptors(feats, gaussian_pyramid, intervals + 3, descr_width, descr_hist_bins);

	__feats2KeyPoints(feats, keypoints);

	__featsVec2Mat(feats, descriptor);
}

void __feats2KeyPoints(const vector<Feature> &feats, vector<KeyPoint> &keypoints) {
	int n = feats.size();
	keypoints.reserve(n);
	for (int i = 0; i < n; i ++) {
		const Feature &feat = feats[i];
		keypoints.push_back(KeyPoint(feat.__x, feat.__y, SIFT_KEYPOINT_DIAMETER * feat.__scl, feat.__ori,
		                             abs(feat.__contrast), feat.__octave));
	}
}

void __featsVec2Mat(const vector<Feature> &feats, Mat &mat) {
	int row = feats.size();
	int col = feats[0].__descriptor.size();

	mat = Mat(row, col, CV_8UC1);
	for (int r = 0; r < row; r ++) {
		for (int c = 0; c < col; c ++) {
			mat.at<unsigned char>(r, c) = feats[r].__descriptor[c];
		}
	}
}

Mat __createInitImg(const Mat &img, bool img_dbl, double sigma) {
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

void __buildGaussPyramid(const Mat &base, vector<Mat> &gaussian_pyramid, int octaves, int intervals, double sigma) {
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

void __buildDogPyramid(const vector<Mat> &gaussian_pyramid, vector<Mat> &dog_pyramid, int octaves, int intervals) {
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

void __scaleSpaceExtrema(const vector<Mat> &dog_pyramid, vector<Feature> &feats, int octaves, int intervals,
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

bool __isExtremum(const vector<Mat> &dog_pyramid, int idx, int r, int c) {
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

bool __interpExtremum(const vector<Mat> &dog_pyramid, Feature &feat, int idx, int r, int c, int intervals, double contrast_thres) {
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
	feat.__contrast = contrast;

	feat.__r = r;
	feat.__c = c;
	feat.__octave = octave;
	feat.__interval = idx % layer_per_octave_dog;
	feat.__idx = idx;
	feat.__sub_interval = xi;

	return true;
}

void __interpStep(const vector<Mat> &dog_pyramid, int idx, int r, int c, double &xi, double &xr, double &xc) {
	Mat dD = __derivative(dog_pyramid, idx, r, c);
	Mat H = __hessian(dog_pyramid, idx, r, c);
	Mat H_inv = H.inv(DECOMP_SVD);
	Mat X = - H_inv * dD;

	xi = X.at<double>(2, 0);
	xr = X.at<double>(1, 0);
	xc = X.at<double>(0, 0);
}

Mat __derivative(const vector<Mat> &dog_pyramid, int idx, int r, int c) {
	Mat dI(3, 1, CV_64FC1);
	dI.at<double>(0, 0) = (dog_pyramid[idx].at<float>(r, c + 1) - dog_pyramid[idx].at<float>(r, c - 1)) / 2.0;
	dI.at<double>(1, 0) = (dog_pyramid[idx].at<float>(r + 1, c) - dog_pyramid[idx].at<float>(r - 1, c)) / 2.0;
	dI.at<double>(2, 0) = (dog_pyramid[idx + 1].at<float>(r, c) - dog_pyramid[idx - 1].at<float>(r, c)) / 2.0;

	return dI;
}

Mat __hessian(const vector<Mat> &dog_pyramid, int idx, int r, int c) {
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

double __interpContrast(const vector<Mat> &dog_pyramid, int idx, int r, int c, double xi, double xr, double xc) {
	Mat dD = __derivative(dog_pyramid, idx, r, c);
	Mat X(3, 1, CV_64FC1);
	X.at<double>(2, 0) = xi;
	X.at<double>(1, 0) = xr;
	X.at<double>(0, 0) = xc;

	Mat t = dD.t() * X;
	return dog_pyramid[idx].at<float>(r, c) + t.at<double>(0, 0) * 0.5;
}

bool __isTooEdgeLike(const Mat &dog, int r, int c, int curvature_thres) {
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

void __calcFeatureScales(vector<Feature> &feats, double sigma, int intervals) {
	for (auto it = feats.begin(); it != feats.end(); it ++) {
		float interval = (*it).__interval + (*it).__sub_interval;
		(*it).__scl = sigma * pow(2.0, (*it).__octave + interval / intervals);
		(*it).__scl_octave = sigma * pow(2.0, interval / intervals);
	}
}

void __adjustForImgDbl(vector<Feature> &feats) {
	for (auto it = feats.begin(); it != feats.end(); it ++) {
		(*it).__x /= 2.0;
		(*it).__y /= 2.0;
		(*it).__scl /= 2.0;
	}
}

void __calcFeatureOris(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave) {
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

void __oriHist(const Mat &gaussian, vector<double> &hist, int r, int c, int rad, double sigma) {
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

bool __calcGradMagOri(const Mat &gaussian, int r, int c, double &mag, double &ori) {
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

void __smoothOriHist(vector<double> &hist) {
	int n = hist.size();
	for (int i = 0; i < n; i++ ) {
		hist[i] = 0.25 * hist[(i + n - 1) % n] + 0.5 * hist[i] + 0.25 * hist[(i + 1) % n];
	}
}

void __addGoodOriFeatures(queue<Feature> &feat_queue, const vector<double> &hist, double mag_thres, const Feature &feat) {
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

void __computeDescriptors(vector<Feature> &feats, const vector<Mat> &gaussian_pyramid, int layer_per_octave, int d, int n) {
	vector<double> hist(d * d * n);
	for (auto it = feats.begin(); it != feats.end(); it ++) {
		for (int i = 0; i < d * d * n; i ++)
			hist[i] = 0;

		Feature &feat = (*it);
		__descriptorHist(gaussian_pyramid[feat.__octave * layer_per_octave + feat.__interval], hist,
		                 feat.__r, feat.__c, feat.__ori, feat.__scl_octave, d, n);
		__hist2Descriptor(hist, feat, d, n);
	}
}

void __descriptorHist(const Mat &gaussian, vector<double> &hist, int r, int c, double ori, double scl, int d, int n) {
	double PI_2 = 2.0 * CV_PI;

	double cos_t = cos( ori );
	double sin_t = sin( ori );
	double bins_per_rad = n / PI_2;
	double exp_denom = d * d * 0.5;
	double hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;

	double g_mag, g_ori;
	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++) {
			double c_rot = ( j * cos_t - i * sin_t ) / hist_width;
			double r_rot = ( j * sin_t + i * cos_t ) / hist_width;
			double rbin = r_rot + d / 2 - 0.5;
			double cbin = c_rot + d / 2 - 0.5;

			if (rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d) {
				if (__calcGradMagOri(gaussian, r + i, c + j, g_mag, g_ori)) {
					g_ori -= ori;
					while ( g_ori < 0.0 )
						g_ori += PI_2;
					while ( g_ori >= PI_2 )
						g_ori -= PI_2;

					double obin = g_ori * bins_per_rad;
					double w = exp(-(c_rot * c_rot + r_rot * r_rot) / exp_denom);
					__interpHistEntry(hist, rbin, cbin, obin, g_mag * w, d, n);
				}
			}
		}
	}
}

void __interpHistEntry(vector<double> &hist, double rbin, double cbin, double obin, double mag, int d, int n) {
	int r0 = floor( rbin );
	int c0 = floor( cbin );
	int o0 = floor( obin );
	double d_r = rbin - r0;
	double d_c = cbin - c0;
	double d_o = obin - o0;

	for (int r = 0; r <= 1; r++) {
		int rb = r0 + r;
		if ( rb < 0  ||  rb >= d )
			continue;

		double v_r = mag * ( ( r == 0 ) ? 1.0 - d_r : d_r );
		for (int c = 0; c <= 1; c++) {
			int cb = c0 + c;
			if ( cb < 0  ||  cb >= d )
				continue;

			double v_c = v_r * ( ( c == 0 ) ? 1.0 - d_c : d_c );
			for (int o = 0; o <= 1; o ++) {
				int ob = ( o0 + o ) % n;
				double v_o = v_c * ( ( o == 0 ) ? 1.0 - d_o : d_o );
				hist[rb * n * d + cb * n + ob] += v_o;
			}
		}
	}
}

void __hist2Descriptor(const vector<double> &hist, Feature &feat, int d, int n) {
	vector<double> descriptor(d * d * n);
	int idx = 0;
	for (int r = 0; r < d; r++) {
		for (int c = 0; c < d; c++) {
			for (int o = 0; o < n; o++) {
				descriptor[idx] = hist[idx];
				idx ++;
			}
		}
	}

	__normalizeDescriptor(descriptor);
	for (size_t i = 0; i < descriptor.size(); i++) {
		if (descriptor[i] > SIFT_DESCR_MAG_THR)
			descriptor[i] = SIFT_DESCR_MAG_THR;
	}
	__normalizeDescriptor(descriptor);

	feat.__descriptor.reserve(d * d * n);
	for (size_t i = 0; i < descriptor.size(); i++) {
		int int_val = SIFT_INT_DESCR_FCTR * descriptor[i];
		feat.__descriptor.push_back(min(255, int_val));
	}
}

void __normalizeDescriptor(vector<double> &descriptor) {
	double len_sq = 0;
	for (size_t i = 0; i < descriptor.size(); i++) {
		double cur = descriptor[i];
		len_sq += cur * cur;
	}
	double len_inv = 1.0 / sqrt( len_sq );
	for (size_t i = 0; i < descriptor.size(); i++)
		descriptor[i] *= len_inv;
}
#include <opencv2/nonfree/features2d.hpp>

#include "SiftHelper.hpp"

void siftWrapper(const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptor) {
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	gray_img.convertTo(gray_img, CV_32FC1, 1.0 / 255);

	extractSiftFeatures(gray_img, keypoints, descriptor);
}

void match2img(const char *img1_p, const char *img2_p, Mat &output) {
	Mat img1 = imread(img1_p);
	vector<KeyPoint> keypoints1;
	Mat descriptor1;

	Mat img2 = imread(img2_p);
	vector<KeyPoint> keypoints2;
	Mat descriptor2;

	siftWrapper(img1, keypoints1, descriptor1);
	siftWrapper(img2, keypoints2, descriptor2);

	/***************************************OPENCV*******************************/
	SiftFeatureDetector detector;
	SiftDescriptorExtractor extractor;

	// detector.detect(img1, keypoints1);
	// extractor.compute(img1, keypoints1, descriptor1);

	// detector.detect(img2, keypoints2);
	// extractor.compute(img2, keypoints2, descriptor2);
	/***************************************END***********************************/

	BFMatcher matcher;
	vector<vector<DMatch> > matches;
	matcher.knnMatch(descriptor1, descriptor2, matches, 2);

	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i ++) {
		if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
			good_matches.push_back(matches[i][0]);
		}
	}

	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output, Scalar::all(-1), Scalar::all(-1),
	            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}
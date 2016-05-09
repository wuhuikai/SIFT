#include <opencv2/nonfree/features2d.hpp>

#include "SiftHelper.hpp"

int main() {
	// /*****************************KEY_POINTS********************************/
	// Mat img = imread("imgs/lovely.jpg");

	// //MINE
	// vector<KeyPoint> keypoints;
	// Mat descriptor;
	// siftWrapper(img, keypoints, descriptor);

	// Mat my_sift;
	// drawKeypoints(img, keypoints, my_sift, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// cout << "MINE:" << keypoints.size() << endl;
	// imshow("MINE-SIFT", my_sift);
	// imwrite("SIFT.jpg", my_sift);

	// // OPENCV
	// SiftFeatureDetector detector;
	// vector<KeyPoint> keypoints_sift;
	// detector.detect(img, keypoints_sift);
	// Mat output;
	// drawKeypoints(img, keypoints_sift, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// cout << "Opencv:" << keypoints_sift.size() << endl;
	// imshow("OPEN-SIFT", output);

	/*****************************MATCH*************************************/
	Mat match_out;
	match2img("imgs/1.jpg", "imgs/2.jpg", match_out);
	imshow("MATCH", match_out);
	imwrite("MATCH.jpg", match_out);

	waitKey();
	return 0;
}
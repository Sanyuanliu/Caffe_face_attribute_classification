//celeba aligment 

#include <fstream>
#include<iostream>
#include<vector>
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;

cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv = Mat()) {
	assert(source_points.size() == target_points.size());
	assert(source_points.size() >= 2);
	Mat U = Mat::zeros(target_points.size() * 2, 1, CV_64F);
	Mat X = Mat::zeros(source_points.size() * 2, 4, CV_64F);
	for (int i = 0; i < target_points.size(); i++) {
		U.at<double>(i * 2, 0) = source_points[i].x;
		U.at<double>(i * 2 + 1, 0) = source_points[i].y;
		X.at<double>(i * 2, 0) = target_points[i].x;
		X.at<double>(i * 2, 1) = target_points[i].y;
		X.at<double>(i * 2, 2) = 1;
		X.at<double>(i * 2, 3) = 0;
		X.at<double>(i * 2 + 1, 0) = target_points[i].y;
		X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
		X.at<double>(i * 2 + 1, 2) = 0;
		X.at<double>(i * 2 + 1, 3) = 1;
	}
	Mat r = X.inv(DECOMP_SVD)*U;
	Tinv = (Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
		r.at<double>(1), r.at<double>(0), 0,
		r.at<double>(2), r.at<double>(3), 1);
	Mat T = Tinv.inv(DECOMP_SVD);
	Tinv = Tinv(Rect(0, 0, 2, 3)).t();
	return T(Rect(0, 0, 2, 3)).t();
}
cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv = Mat()) {
	Mat Tinv1, Tinv2;
	Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
	std::vector<Point2d> source_point_reflect;
	for (auto sp : source_points) {
		source_point_reflect.push_back(Point2d(-sp.x, sp.y));
	}
	swap(source_point_reflect[0], source_point_reflect[1]);
   	swap(source_point_reflect[3], source_point_reflect[4]);
	Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
	trans2.colRange(0, 1) *= -1;
	Tinv2.rowRange(0, 1) *= -1;
	std::vector<Point2d> trans_points1, trans_points2;
	transform(source_points, trans_points1, trans1);
	transform(source_points, trans_points2, trans2);
	swap(trans_points2[0], trans_points2[1]);
    	swap(trans_points2[3], trans_points2[4]);
	double norm1 = norm(Mat(trans_points1), Mat(target_points), NORM_L2);
	double norm2 = norm(Mat(trans_points2), Mat(target_points), NORM_L2);
	Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
	return norm1 < norm2 ? trans1 : trans2;
}



int main() {
	ifstream label_attribute("D:\\DataSet\\CelebA\\list_landmarks_celeba.txt");
	string img_dir = "D:\\DataSet\\CelebA\\Img\\img_celeba.7z\\img_celeba.7z\\img_celeba\\";
	string dstimg_dir = "E:\\Dataset\\crop_by_me\\";
	string point_attribute;
	vector<Point2d> target_points = { {98.4,  102.4 },{ 147.7,  102.1 },{ 123.2,  103.4 },{ 102.97,159.3 },{ 143.8,159.1} };

	Mat trans_inv;
	std::vector<cv::Point2d> points;
	Mat trans;
	Mat cropImage;
	Mat image;

	while (getline(label_attribute, point_attribute))
	{
		string buf;
		stringstream ss(point_attribute);
		vector<string> tokens;
		while (ss >> buf) {
			tokens.push_back(buf);
		}

		for (int i = 1; i < 6; i++) {
			points.push_back(cv::Point2d(std::stoi(tokens[2 * i - 1]), std::stoi(tokens[2 * i])));
			}
		trans = findSimilarityTransform(points,target_points, trans_inv);
		image = imread(img_dir + tokens[0]);
		warpAffine(image, cropImage, trans, Size(256, 256));
		//imshow("Q", cropImage);
		//waitKey(1);
		imwrite(dstimg_dir+tokens[0],  cropImage);
		points.clear();
	}

	return 0;
}

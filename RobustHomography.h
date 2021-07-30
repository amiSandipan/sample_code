#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d.hpp>
#include<opencv2\opencv_modules.hpp>

using namespace std;
using namespace cv;

class RobustHomography
{
	Mat image1, image2;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;
public:
	RobustHomography(Mat &, Mat &);
	void computeHomography();
	virtual ~RobustHomography();
};


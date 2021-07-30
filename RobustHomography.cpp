#include "stdafx.h"
#include<iostream>
#include "RobustHomography.h"


RobustHomography::RobustHomography(Mat & i1, Mat & i2)
{
	image1 = i1;
	image2 = i2;
	detector = ORB::create();
	extractor = ORB::create();
	matcher = BFMatcher::create(NORM_HAMMING, false);

}

void RobustHomography::computeHomography()
{
	//Extract KeyPoints
	std::vector<cv::KeyPoint> ORBKeypoints1, ORBKeypoints2;
	detector->detect(image1, ORBKeypoints1);
	detector->detect(image2, ORBKeypoints2);

	//Compute Descriptors
	Mat descriptor1, descriptor2;
	extractor->compute(image1, ORBKeypoints1, descriptor1);
	extractor->compute(image2, ORBKeypoints2, descriptor2);

	//Compute Matches between Keypoints
	vector< DMatch > matches;
	vector< vector<DMatch> > knnMatches;
	matcher->knnMatch(descriptor2, descriptor1, knnMatches, 2);

	//Validate Matches [Ratio Test]
	std::vector<Point2f> match1, match2;
	const float minRatio = 0.75;
	for (auto knnMatch : knnMatches)
	{
		const cv::DMatch& bestMatch = knnMatch[0];
		const cv::DMatch& betterMatch = knnMatch[1];
		float distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
		{
			matches.push_back(bestMatch);
		}
	}

	for (auto match : matches)
	{
		match2.push_back(ORBKeypoints2[match.queryIdx].pt);
		match1.push_back(ORBKeypoints1[match.trainIdx].pt);
	}

	//Compute Homegraphy and Warp Perspective
	Mat homographyMatrix = findHomography(match2, match1, CV_RANSAC);
	std::cout << "Homography Matrix - \n" << homographyMatrix << std::endl;
	Mat warpedImage;
	warpPerspective(image2, warpedImage, homographyMatrix, image2.size());

	//Concatenate Images for Output
	putText(image1, "Image1", Point2d(50, image1.rows - 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 4, 20);
	putText(image2, "Image2", Point2d(50, image2.rows - 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 4, 20);
	putText(warpedImage, "Image 2 Warped Into Image 1 Perspective", Point2d(50, warpedImage.rows - 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 4, 20);

	//Concatenate Images of Final Result
	Mat tempConcatedImages, finalConcatenatedImages;
	hconcat(image1, warpedImage, tempConcatedImages);
	hconcat(tempConcatedImages, image2, finalConcatenatedImages);
	namedWindow("Final", WINDOW_NORMAL);// Create a window for display.
	imshow("Final", finalConcatenatedImages);
}


RobustHomography::~RobustHomography()
{
}

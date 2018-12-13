#ifndef _ICP_H_
#define _ICP_H_

#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "iostream"

void findClosestPointSet(std::vector<cv::Point3d>& model, std::vector<cv::Point3d>& data, std::vector<cv::Point3d>& Y);

void ICP(std::vector<cv::Point3d>& model, std::vector<cv::Point3d>& data, double *R, double *T, double e);

void calcCenterPoint(std::vector<cv::Point3d>& Points, cv::Point3d& _mean_P);



#endif // _ICP_H_

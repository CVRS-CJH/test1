#include "ICP.h"
#include "time.h"

void ICP(std::vector<cv::Point3d>& model, std::vector<cv::Point3d>& data, double *R, double *T, double e)
{
	std::vector<cv::Point3d> Y, P;
	std::vector<cv::Point3d>::iterator itr1, itr2;
	double pre_d = 0.0, d = 0.0;
	int round = 0;
	P = data;
	do 
	{
		pre_d = d;

		clock_t start_time1 = clock();
		findClosestPointSet(model, P, Y);
		clock_t end_time1 = clock();
		std::cout << "findClosestPointSet time: " << (double)(end_time1 - start_time1) / CLOCKS_PER_SEC << std::endl;
		cv::Point3d _mean_P, _mean_Y;
		calcCenterPoint(P, _mean_P);
		calcCenterPoint(Y, _mean_Y);

		


	} while (abs(pre_d - d)<=e);

}
void findClosestPointSet(std::vector<cv::Point3d>& model, std::vector<cv::Point3d>& data, std::vector<cv::Point3d>& Y)
{
	std::vector<cv::Point3d>::iterator itr1, itr2;
	Y.clear();
	for (itr1= data.begin();itr1<data.end();itr1++)
	{
		double min;
		std::vector<cv::Point3d>::iterator it;
		itr2 = model.begin();
		min = (itr1->x - itr2->x)*(itr1->x - itr2->x) + (itr1->y - itr2->y)*(itr1->y - itr2->y) + (itr1->z - itr2->z)*(itr1->z - itr2->z);
		itr2++;
		for (;itr2 != model.end();itr2++)
		{
			double d;
			d = (itr1->x - itr2->x)*(itr1->x - itr2->x) + (itr1->y - itr2->y)*(itr1->y - itr2->y) + (itr1->z - itr2->z)*(itr1->z - itr2->z);
			if (d<min)
			{
				min = d;
				it = itr2;
			}
		}

		Y.push_back(*it);
	}
}

void calcCenterPoint(std::vector<cv::Point3d>& Points, cv::Point3d& _mean_P)
{
	cv::Point3d p_mean;
	int N = Points.size();
	for (int i = 0;i<N; i++)
	{
		p_mean += Points[i];
	}
	p_mean.x /= N;
	p_mean.y /= N;
	p_mean.z /= N;

	_mean_P = p_mean;
}
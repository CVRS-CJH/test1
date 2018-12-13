#include <iostream>
#include "optimizer.h"
#include "ICP-src.h"
#include <math.h>
#include <algorithm>
#include "sophus/se3.hpp"
#include <mutex>
#include <thread>
#include <windows.h>

#include <opencv2/opencv.hpp>
#include "opencv2/ml.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <iterator>
#include <array>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std;
using namespace cv;


typedef std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef std::vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 9, 1> Vector9d;

double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

void detectMatches(const cv::Mat& img1, const cv::Mat& img2, VecVector2d& p2d, VecVector3d& p3d)
{
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::DMatch> matches;
	cv::Mat descriptor1, descriptor2;

	
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create("ORB");
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	descriptor->compute(img1, keypoints1, descriptor1);
	descriptor->compute(img2, keypoints2, descriptor2);

	std::vector<cv::DMatch> match;
	matcher->match(descriptor1, descriptor2, match);

	double min_dsit = 10000, max_dist = 0;
	for (int i = 0; i < descriptor1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dsit)
		{
			min_dsit = dist;
		}
		if (dist > max_dist)
		{
			max_dist = dist;
		}

	}
	std::cout << "最大距离： " << max_dist << std::endl;
	std::cout << "最小距离： " << min_dsit << std::endl;

	for (int i = 0; i < descriptor1.rows; i++)
	{
		if (match[i].distance <= std::max(2 * min_dsit, 30.0))
		{
			matches.push_back(match[i]);
		}
	}

	std::cout << "一共找到" << matches.size() << "对匹配点" << std::endl;

	cv::Mat d1 = cv::imread("D:\\Data\\1_depth.png", CV_LOAD_IMAGE_UNCHANGED);

	for (auto m : matches)
	{
		ushort d = d1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
		if (d == 0)
		{
			continue;
		}
		float z = d / 5000.0;
		float x = (keypoints1[m.queryIdx].pt.x - cx)*z / fx;
		float y = (keypoints1[m.queryIdx].pt.y - cy)*z / fy;
		p3d.push_back(Eigen::Vector3d(x, y, z));
		p2d.push_back(Eigen::Vector2d(keypoints2[m.trainIdx].pt.x, keypoints2[m.trainIdx].pt.y));
	}

}
void findCorrespondPoints(const cv::Mat& img1, const cv::Mat& img2, VecVector2d& p2d1, VecVector2d& p2d2, VecVector3d& p3d)
{
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::DMatch> matches;
	cv::Mat descriptor1, descriptor2;

	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create("OrbDetector");
	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create("OrbDescriptor");
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	descriptor->compute(img1, keypoints1, descriptor1);
	descriptor->compute(img2, keypoints2, descriptor2);

	std::vector<cv::DMatch> match;
	matcher->match(descriptor1, descriptor2, match);

	double min_dsit = 10000, max_dist = 0;
	for (int i = 0; i < descriptor1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dsit)
		{
			min_dsit = dist;
		}
		if (dist > max_dist)
		{
			max_dist = dist;
		}

	}
	std::cout << "最大距离： " << max_dist << std::endl;
	std::cout << "最小距离： " << min_dsit << std::endl;

	for (int i = 0; i < descriptor1.rows; i++)
	{
		if (match[i].distance <= std::max(2 * min_dsit, 30.0))
		{
			matches.push_back(match[i]);
		}
	}

	std::cout << "一共找到" << matches.size() << "对匹配点" << std::endl;

	cv::Mat d1 = cv::imread("D:\\Data\\1_depth.png", CV_LOAD_IMAGE_UNCHANGED);

	for (auto m : matches)
	{
		ushort d = d1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
		if (d == 0)
		{
			continue;
		}
		float z = d / 5000.0;
		float x = (keypoints1[m.queryIdx].pt.x - cx)*z / fx;
		float y = (keypoints1[m.queryIdx].pt.y - cy)*z / fy;
		p3d.push_back(Eigen::Vector3d(x, y, z));
		p2d1.push_back(Eigen::Vector2d(keypoints1[m.queryIdx].pt.x, keypoints2[m.queryIdx].pt.y));
		p2d2.push_back(Eigen::Vector2d(keypoints2[m.trainIdx].pt.x, keypoints2[m.trainIdx].pt.y));
	}

}
cv::Mat toCVMat(g2o::SE3Quat & SE3)
{
	cv::Mat pose(4, 4, CV_64F);
	Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();

	for (size_t i = 0; i < 4; i++)
	{
		for (size_t j = 0; j < 4; j++)
		{
			pose.at<double>(i, j) = eigMat(i, j);
		}
	}

	return pose.clone();


}

//训练一 *******Gauss Newton 拟合曲线********//

//obj = A * sin(Bx) + C * cos(D*x) - F

//const double DERIV_STEP = 1e-5;
//const int MAX_ITER = 100;
//
//double func(const Eigen::VectorXd &input, const Eigen::VectorXd &output, const Eigen::VectorXd &params, double objindex)
//{
//	double A = params(0);
//	double B = params(1);
//	double C = params(2);
//	double D = params(3);
//
//	double x = input(objindex);
//	double y = output(objindex);
//
//	double error = A * sin(B*x) + C * cos(D*x) - y;
//	return error;
//}
//Eigen::VectorXd objF(const Eigen::VectorXd &input, const Eigen::VectorXd &output, const Eigen::VectorXd &params)
//{
//
//	Eigen::VectorXd obj(input.rows());
//	for (int i = 0; i<input.rows();i++)
//	{
//		obj(i) = func(input, output, params, i);
//	}
//
//	return obj;
//
//}
//double ComputeError(Eigen::VectorXd &obj)
//{
//	return obj.squaredNorm() / 2;
//
//}
//double compute_jac(const Eigen::VectorXd &input, const Eigen::VectorXd &output, int obj_index,int params_index, const Eigen::VectorXd &params)
//{
//	Eigen::VectorXd params_delta1 = params;
//	Eigen::VectorXd params_delta2 = params;
//
//	params_delta1(params_index) -= DERIV_STEP;
//	params_delta2(params_index) += DERIV_STEP;
//
//	double obj1 = func(input, output, params_delta1, obj_index);
//	double obj2 = func(input, output, params_delta2, obj_index);
//
//	return (obj2 - obj1) / (2 * DERIV_STEP);
//}
//
//Eigen::MatrixXd Jacobin(const Eigen::VectorXd &input, const Eigen::VectorXd &output, const Eigen::VectorXd &params)
//{
//	int jacrows = input.rows();
//	int jaccols = params.rows();
//
//	Eigen::MatrixXd Jac(jacrows, jaccols);
//	for (int i=0; i< jacrows;i++)
//	{
//		for (int j = 0; j<jaccols;j++)
//		{
//			Jac(i, j) = compute_jac(input, output,i,j,params);
//		}
//	}
//
//	return Jac;
//}
//void gaussNewton(const Eigen::VectorXd &input, const Eigen::VectorXd &output, Eigen::VectorXd &params)
//{
//	int errornum = input.rows();
//	int paramsnum = params.rows();
//
//	Eigen::VectorXd obj(errornum);
//	int iterCnt = 0;
//	double last_errorsum = 0;
//
//	while (iterCnt < MAX_ITER)
//	{
//		obj = objF(input, output, params);
//		//std::cout << "obj:" << std::endl << obj << std::endl;
//		double error_sum = 0;
//		error_sum = ComputeError(obj);
//
//		std::cout << "Interator index: " << iterCnt << std::endl;
//		std::cout << "parameter: " <<std::endl<< params << std::endl;
//		std::cout << "error: " << error_sum << std::endl;
//		
//
//		if (fabs(error_sum - last_errorsum) <= 1e-12)
//		{
//			break;
//		}
//		last_errorsum = error_sum;
//		//计算雅克比矩阵
//		Eigen::MatrixXd Jac = Jacobin(input,output,params);
//
//		Eigen::VectorXd delta(paramsnum);
//
//		delta = -(Jac.transpose()*Jac).inverse() * Jac.transpose()*obj;
//		params += delta;
//		iterCnt++;
//	}
//
//	
//}
//
//double maxMatrixDiagonale(const MatrixXd& Hessian)
//{
//	int max = 0;
//	for (int i = 0; i < Hessian.rows(); i++)
//	{
//		if (Hessian(i, i) > max)
//			max = Hessian(i, i);
//	}
//
//	return max;
//}
//
////L(h) = F(x) + h^t*J^t*f + h^t*J^t*J*h/2
////deltaL = h^t * (u * h - g)/2
//double linerDeltaL(const VectorXd& step, const VectorXd& gradient, const double u)
//{
//	double L = step.transpose() * (u * step - gradient);
//	return L / 2;
//}
//
//void levenMar(const VectorXd& input, const VectorXd& output, VectorXd& params)
//{
//	int errNum = input.rows();      //error num
//	int paraNum = params.rows();    //parameter num
//
//									//initial parameter 
//	VectorXd obj = objF(input, output, params);
//	MatrixXd Jac = Jacobin(input, output, params);  //jacobin
//	MatrixXd A = Jac.transpose() * Jac;             //Hessian
//	VectorXd gradient = Jac.transpose() * obj;      //gradient
//
//													//initial parameter tao v epsilon1 epsilon2
//	double tao = 1e-3;
//	long long v = 2;
//	double eps1 = 1e-12, eps2 = 1e-12;
//	double u = tao * maxMatrixDiagonale(A);
//	bool found = gradient.norm() <= eps1;
//	if (found) return;
//
//	double last_sum = 0;
//	int iterCnt = 0;
//
//	while (iterCnt < MAX_ITER)
//	{
//		VectorXd obj = objF(input, output, params);
//
//		MatrixXd Jac = Jacobin(input, output, params);  //jacobin
//		MatrixXd A = Jac.transpose() * Jac;             //Hessian
//		VectorXd gradient = Jac.transpose() * obj;      //gradient
//
//		if (gradient.norm() <= eps1)
//		{
//			cout << "stop g(x) = 0 for a local minimizer optimizer." << endl;
//			break;
//		}
//
//		cout << "A: " << endl << A << endl;
//
//		VectorXd step = (A + u * MatrixXd::Identity(paraNum, paraNum)).inverse() * gradient; //negtive Hlm.
//
//		cout << "step: " << endl << step << endl;
//
//		if (step.norm() <= eps2*(params.norm() + eps2))
//		{
//			cout << "stop because change in x is small" << endl;
//			break;
//		}
//
//		VectorXd paramsNew(params.rows());
//		paramsNew = params - step; //h_lm = -step;
//
//								   //compute f(x)
//		obj = objF(input, output, params);
//
//		//compute f(x_new)
//		VectorXd obj_new = objF(input, output, paramsNew);
//
//		double deltaF = ComputeError(obj) - ComputeError(obj_new);
//		double deltaL = linerDeltaL(-1 * step, gradient, u);
//
//		double roi = deltaF / deltaL;
//		cout << "roi is : " << roi << endl;
//		if (roi > 0)
//		{
//			params = paramsNew;
//			u *= max(1.0 / 3.0, 1 - pow(2 * roi - 1, 3));
//			v = 2;
//		}
//		else
//		{
//			u = u * v;
//			v = v * 2;
//		}
//
//		cout << "u = " << u << " v = " << v << endl;
//
//		iterCnt++;
//		cout << "Iterator " << iterCnt << " times, result is :" << endl << endl;
//	}
//}
//int main(int argc,char *argv[])
//{
//	A x(4,5);
//	std::cout << "x.a x.b" << x.a << " " << x.b << std::endl;
//	int num_params = 4;
//
//	int total_data = 100;
//	Eigen::VectorXd input(total_data);
//	Eigen::VectorXd output(total_data);
//
//	double A = 5, B = 1, C = 10, D = 2;
//	for (int i = 0;i<total_data;i++)
//	{
//		double x =  20.0 * ((rand() % 1000) / 1000.0) - 10.0;
//		double deltay = 2.0 * (rand() % 1000) / 1000.0;
//		double y = A * sin(B*x) + C * cos(D*x) + deltay;
//		//std::cout << "deltay: " << deltay << std::endl;
//
//		input(i) = x;
//		output(i) = y;
//	}
//
//	Eigen::VectorXd params_gaussNewton(num_params);
//	params_gaussNewton << 1.6, 1.4, 6.2, 1.7;
//	//params_gaussNewton << 4, 0.9, 8, 1;
//	std::cout <<"input"<< input << std::endl;
//	std::cout << "output" << output << std::endl;
//	VectorXd params_levenMar = params_gaussNewton;
//
//	levenMar(input, output, params_levenMar);
//	gaussNewton(input,output,params_gaussNewton);
//
//	std::cout << "gauss newton parameter: " << std::endl << params_gaussNewton << std::endl;
//	std::cout << "levenMar parameter: " << std::endl << params_levenMar << std::endl;
//
//	system("pause");
//}

//训练2   g2o曲线拟合******//

//int main()
//{
//	double a = 5.0, b = 1.0, c = 10.0, d = 2.0; // 真实参数值
//	int N = 100;
//
//	double w_sigma = 2.0;
//	cv::RNG rng;
//	double abcd[4] = { 0,0,0,0 };
//	vector<double> x_data, y_data;
//
//	for (int i = 0;i<N;i++)
//	{
//		double x = rng.uniform(-10., 10.);
//		double y = a * sin(b*x) + c * cos(d *x) + rng.gaussian(w_sigma);
//		x_data.push_back(x);
//		y_data.push_back(y);
//
//		cout << x_data[i] << " , " << y_data[i] << endl;
//	}
//
//
//	// 构建图优化，先设定g2o
//	// 矩阵块：每个误差项优化变量维度为4 ，误差值维度为1
//	typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 1> > Block;
//	//// 线性方程求解器：稠密的增量方程
//	Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
//
//	Block* solver_ptr = new Block(linearSolver);    // 矩阵块求解器
//
//													// 梯度下降方法，从GN, LM, DogLeg 中选
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
//	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
//
//	g2o::SparseOptimizer optimizer;     // 图模型
//	optimizer.setAlgorithm(solver);   // 设置求解器
//	optimizer.setVerbose(true);     // 打开调试输出
//
//									// 往图中增加顶点
//
//	//typedef g2o::BlockSolver<g2o::BlockSolverTraits<4, 1>> Block;
//	//Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
//	//Block* solver_ptr = new Block(linearSolver);
//	//g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//
//	/*g2o::SparseOptimizer optimizer;
//	optimizer.setAlgorithm(solver);
//	optimizer.setVerbose(true);
//*/
//	CurveFittingVertex *v = new CurveFittingVertex();
//	// 设置优化初始估计值
//	v->setEstimate(Eigen::Vector4d(1.6, 1.4, 6.2, 1.7));
//	v->setId(0);
//	v->setFixed(false);
//
//	// 往图中增加边
//	for (int i = 0; i < N; i++)
//	{
//		CurveFittingEdge* edge = new CurveFittingEdge();
//		edge->setId(i + 1);
//		edge->setVertex(0, v);      // 设置连接的顶点
//		edge->setMeasurement(y_data[i]);      // 观测数值
//
//									  // 信息矩阵：协方差矩阵之逆
//		edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma* w_sigma));
//
//		edge->_x = x_data[i];
//
//		optimizer.addEdge(edge);
//	}
//
//	// 执行优化
//	cout << "strat optimization" << endl;
//
//	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//
//	optimizer.initializeOptimization();
//	optimizer.optimize(100);
//
//	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//
//	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
//	cout << "solve time cost = " << time_used.count() << " seconds." << endl;
//
//	// 输出优化值
//	Eigen::Vector4d abcd_estimate = v->estimate();
//	cout << "estimated module: " << endl << abcd_estimate << endl;
//
//	return 0;
//}

//
//
//// 待优化变量——曲线模型的顶点，模板参数：优化变量维度　和　数据类型
//class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>//定点类型  a b c三维变量
//{
//public:
//	EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 类成员 有Eigen  变量时需要 显示 加此句话 宏定义
//		virtual void setToOriginImpl() // 虚函数 重置
//	{
//		_estimate << 0, 0, 0;// 初始化定点  优化变量值初始化
//	}
//	virtual void oplusImpl(const double* update) // 更新
//	{
//		_estimate += Eigen::Vector3d(update);//迭代更新 变量
//	}
//	//虚函数  存盘和读盘：留空
//	virtual bool read(istream& in) { return true; }
//	virtual bool write(ostream& out) const { return true; }
//};
//
//
//// 误差模型—— 曲线模型的边, 模板参数：观测值维度(输入的参数维度)，类型，连接顶点类型(创建的顶点)
//// 一元边 BaseUnaryEdge<1,double,CurveFittingVertex> 
//// 二元边 BaseBinaryEdge<2,double,CurveFittingVertex>
//// 多元边 BaseMultiEdge<>
//class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>//基础一元 边类型
//{
//public:
//	EIGEN_MAKE_ALIGNED_OPERATOR_NEW// 类成员 有Eigen  变量时需要 显示 加此句话 宏定义
//		CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}//初始化函数   直接赋值  _x = x
//															  // 计算曲线模型误差
//	void computeError()
//	{
//		const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);//顶点
//		const Eigen::Vector3d abc = v->estimate();//获取顶点的优化变量
//		_error(0, 0) = _measurement - std::exp(abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0));//一个误差项 _measurement为测量值
//	}
//	// 存盘和读盘：留空
//	virtual bool read(istream& in) { return true; }
//	virtual bool write(ostream& out) const { return true; }
//public:
//	double _x;  // x 值， y 值为 _measurement
//};
//
//int main(int argc, char** argv)
//{
//	double a = 1.0, b = 2.0, c = 1.0;         // 真实参数值
//	int N = 100;                          // 数据点
//	double w_sigma = 1.0;                 // 噪声Sigma值
//	cv::RNG rng;                        // OpenCV随机数产生器
//	double abc[3] = { 0,0,0 };            // abc参数的估计值
//
//	vector<double> x_data, y_data;      // 数据
//
//	cout << "generating data: " << endl;
//	for (int i = 0; i < N; i++)
//	{
//		double x = i / 100.0;
//		x_data.push_back(x);
//		y_data.push_back(exp(a*x*x + b*x + c) + rng.gaussian(w_sigma));//加上高斯噪声
//		cout << x_data[i] << "\t" << y_data[i] << endl;
//	}
//
//	// 构建图优化解决方案，先设定g2o
//	typedef g2o::BlockSolver< g2o::BlockSolverTraits<3, 1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1
//																	 // 线性方程求解器   H * Δx = −b
//	Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
//																								 // 稀疏矩阵块求解器 用于求解 雅克比J ( 得到右边 b = e转置 *  Ω * J ) 和  海塞矩阵 H  左边 H = J转置* Ω * J   
//	Block* solver_ptr = new Block(linearSolver);      // 矩阵块求解器
//													  // 迭代算法    梯度下降方法，从高斯牛顿GN,  莱文贝格－马夸特方法LM, 狗腿法DogLeg 中选
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
//	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
//	g2o::SparseOptimizer optimizer;     //稀疏 优化模型
//	optimizer.setAlgorithm(solver);   // 设置求解器
//	optimizer.setVerbose(true);       // 打开调试输出
//
//									  // 往图中增加顶点
//	CurveFittingVertex* v = new CurveFittingVertex();//曲线拟合 新建 顶点类型
//	v->setEstimate(Eigen::Vector3d(0, 0, 0));
//	v->setId(0);//id
//	optimizer.addVertex(v);
//
//	// 往图中增加边
//	for (int i = 0; i < N; i++)
//	{
//		CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);//新建 边 带入 观测数据
//		edge->setId(i);//id
//		edge->setVertex(0, v);           // 设置连接的顶点
//		edge->setMeasurement(y_data[i]); // 观测数值
//		edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma*w_sigma)); // 信息矩阵：单位阵协方差矩阵之逆 (个误差项权重)  就一个误差项_error(0,0) 
//		optimizer.addEdge(edge);//添加边
//	}
//
//	// 执行优化
//	cout << "start optimization" << endl;
//	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时
//	optimizer.initializeOptimization();//初始化优化器
//	optimizer.optimize(100);//优化次数
//	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//结束计时
//	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//	cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
//
//	// 输出优化值
//	Eigen::Vector3d abc_estimate = v->estimate();
//	cout << "estimated model: " << abc_estimate.transpose() << endl;
//
//	return 0;
//}

//训练3 重载操作符  //

//class BOX
//{
//public:
//	BOX();
//	~BOX();
//	double getVolume()
//	{
//		return length*height*width;
//	}
//	void setSide(double x,double y,double z)
//	{
//		length = x;
//		width = y;
//		height = z;
//	}
//	BOX operator+(const BOX &b)
//	{
//		BOX box;
//		box.length = b.length + this->length;
//		box.width = b.width + this->width;
//		box.height = b.height + this->height;
//		return box;
//	}
//private:
//	double length;
//	double width;
//	double height;
//};
//
//BOX::BOX()
//{
//}
//
//BOX::~BOX()
//{
//}
//
//int main()
//{
//	BOX a,b,c;
//	a.setSide(10, 10, 10);
//	b.setSide(20, 20, 20);
//	//c.setSide(300, 300, 300);
//	c = a + b;
//	std::cout << "a.volume: " << a.getVolume() << std::endl;
//	std::cout << "a.volume: " << b.getVolume() << std::endl;
//	std::cout << "a.volume: " << c.getVolume() << std::endl;
//
//	system("pause");
//}

//训练4 map自定义排序  //

//typedef struct student
//{
//	int age;
//	double height;
//	double wight;
//}myKey;
//
//typedef struct Mykey
//{
//	std::string mytext;
//}myValue;
//
//struct myComparator
//{
//	bool operator()(const myKey &value1, const myKey &value2) const
//	{
//		if (value1.age != value2.age)
//		{
//			return value1.age < value2.age;
//		}
//		if (value1.height != value2.height)
//		{
//			return value1.height < value2.height;
//		}
//		if (value1.wight != value2.wight)
//		{
//			return value1.wight < value2.wight;
//		}
//
//		return false;
//	}
//};
//int main()
//{
//	std::map<myKey, myValue, myComparator> studentslist;
//
//	myKey v1;
//	v1.age = 10;
//	v1.height = 150;
//	v1.wight = 50;
//	myValue s1;
//	s1.mytext = "v1.age = 10, v1.height = 150, v1.wight = 50";
//
//
//	myKey v2;
//	v2.age = 12;
//	v2.height = 151;
//	v2.wight = 52;
//	myValue s2;
//	s2.mytext = "v2.age = 12, v2.height = 151, v2.wight = 52";
//
//	myKey v3;
//	v3.age = 10;
//	v3.height = 150;
//	v3.wight = 52;
//	myValue s3;
//	s3.mytext = "v3.age = 10, v3.height = 150, v3.wight = 52";
//
//	studentslist[v1] = s1;
//	studentslist[v2] = s2;
//	studentslist[v3] = s3;
//
//	for (auto it = studentslist.begin(); it != studentslist.end(); it++)
//	{
//		std::cout << it->second.mytext.c_str() << " " << std::endl;
//	}
//
//	return 0;
//}

//训练5 合并图像  //

//void creatMatchimage(cv::Mat &img1, cv::Mat&img2, cv::Mat & output)
//{
//	cv::Mat imgleft = img1.clone();
//	cv::Mat imgright = img2.clone();
//	cv::Mat stereoimg;
//
//	int wl, hl, wr, hr, w, h;
//
//	wl = cvRound(imgleft.cols);
//	hl = cvRound(imgleft.rows);
//	wr = cvRound(imgright.cols);
//	hr = cvRound(imgright.rows);
//
//	w = wl + wr;
//	h = hl > hr ? hl : hr;
//	stereoimg.create(h,w,CV_8UC3);
//
//	cv::Mat imgleftRGB,imgrightRGB;
//
//	cv::cvtColor(imgleft, imgleftRGB, CV_GRAY2BGR);
//	cv::cvtColor(imgright, imgrightRGB, CV_GRAY2BGR);
//
//	cv::Mat canvasLeft = stereoimg(cv::Rect(0, 0, wl, h));
//	cv::Mat canvasRight = stereoimg(cv::Rect(wl, 0, wr, h));  //蒙版图像
//	cv::resize(imgleftRGB, canvasLeft, canvasLeft.size(), 0, 0, cv::INTER_AREA);
//	cv::resize(imgrightRGB, canvasRight, canvasRight.size(), 0, 0, cv::INTER_AREA);
//
//	cv::circle(output, cv::Point(100, 100), 3, cv::Scalar(255, 0, 0));
//
//	output = stereoimg.clone();
//
//}
//
//int main()
//{
//	cv::Mat srcImg1 = cv::imread("D:\\Data\\HandHeld-TEST\\rulerimg11\\1\\track\\1.jpg", CV_LOAD_IMAGE_UNCHANGED);
//	cv::Mat srcImg2 = cv::imread("D:\\Data\\HandHeld-TEST\\rulerimg11\\1\\track\\2.jpg", CV_LOAD_IMAGE_UNCHANGED);
//
//	if (srcImg1.channels() == 3)
//	{
//		cv::cvtColor(srcImg1, srcImg1, CV_BGR2GRAY);
//	}
//	if (srcImg2.channels() == 3)
//	{
//		cv::cvtColor(srcImg2, srcImg2, CV_BGR2GRAY);
//	}
//	cv::Mat result;
//	for (int i = 0;i<1000;i++)
//	{
//		std::cout << "i: " << i << std::endl;
//		creatMatchimage(srcImg1, srcImg2,result);
//
//		string name = "D:\\Data\\1\\" + to_string(i) + ".bmp";
//		cv::imwrite(name,result);
//	}
//	
//
//	system("pause");
//	return 0;
//
//
//
//
//}

//训练6 ICP迭代求解RT  //

//void LoadData(std::vector<cv::Point3d>& model,std::vector<cv::Point3d> &data,std::string filename,std::string RTname)
//{
//	std::vector<cv::Point3d> srcData, tranData;
//	std::string line;
//	std::ifstream fp(filename.c_str(),std::ios::in);
//	//while (std::getline(fp,line,'\n'))
//	//{
//	//	if (line.empty())
//	//	{
//	//		continue;
//	//	}
//	//	std::istringstream in(line);
//	//	cv::Point3d point;
//	//	in >> point.x >> point.y >> point.z;
//	//	model.push_back(point);
//	//	
//	//}
//
//	while (!fp.eof())
//	{
//		getline(fp, line);
//		if (!line.empty())
//		{
//			std::stringstream ss;
//			ss << line;
//			cv::Point3d point;
//			ss >> point.x >> point.y >> point.z;
//			model.push_back(point);
//		}
//	}
//
//	std::ifstream fprt (RTname.c_str(), std::ios::in);
//	cv::Mat T;
//	cv::Mat R;
//	cv::Mat t;
//
//	float a[12] = { 0 };
//	fprt >> a[0] >> a[1] >> a[2];
//	fprt >> a[3] >> a[4] >> a[5];
//	fprt >> a[6] >> a[7] >> a[8];
//	fprt >> a[9] >> a[10] >> a[11];
//
//	T = cv::Mat(4, 3, CV_32FC1, a);
//	R = T.rowRange(0, 3);
//	t = T.rowRange(3, 4).t();
//
//	cv::Point3d pointTran;
//	for (int i = 0; i <model.size();i++)
//	{
//		cv::Point3d pointTemp= model[i];
//		//cv::Mat pointMat = cv::Mat(pointTemp);
//		cv::Mat pointMat(3, 1, CV_64FC1);;
//	
//		pointMat.at<double>(0, 0) = pointTemp.x;
//		pointMat.at<double>(1, 0) = pointTemp.y;
//		pointMat.at<double>(2, 0) = pointTemp.z;
//
//		pointMat.convertTo(pointMat, CV_32FC1);
//
//		cv::Mat pointTranMat = R*pointMat + t;
//
//		pointTranMat.convertTo(pointTranMat, CV_64FC1);
//
//		pointTran.x = pointTranMat.at<double>(0, 0);
//		pointTran.y = pointTranMat.at<double>(1, 0);
//		pointTran.z = pointTranMat.at<double>(2, 0);
//
//		data.push_back(pointTran);
//	}
//
//}
//
//void outputData(std::vector<cv::Point3d> &data, std::string pathname)
//{
//	std::ofstream fpout;
//	fpout.open(pathname.c_str());
//	for (size_t i = 0; i < data.size(); i++)
//	{
//		cv::Point3d pTemp = data[i];
//		fpout << pTemp.x << " " << pTemp.y << " " << pTemp.z << "\n";
//	}
//
//}
//
//int main()
//{
//	std::vector<cv::Point3d> model, data;
//	double R[9], T[4], e = 0.0001;
//
//	std::string filename = "D:\\Data\\vlp-0_11_181244_814.txt";
//	std::string RTname = "D:\\Data\\matrix0608Last.txt";
//
//	string outputname = "D:\\Data\\output.txt";
//	string outputname2 = "D:\\Data\\output-src.txt";
//
//	LoadData(model, data, filename,RTname);
//
//	outputData(data, outputname);
//
//	//ICP(model, data, R, T, e);
//	cv::Mat curMat = cv::Mat(model).reshape(1, model.size()).t();
//	cv::Mat prevMat = cv::Mat(data).reshape(1, data.size()).t();
//	cv::Mat RTmp, tTmp;
//	double scale = 1.0;
//	
//
//
//	outputData(data, outputname2);
//
//	std::cout << "end" << std::endl;
//
//	return 0;
//}

 
//训练7 区域生长算法 //

//cv::Mat RegionGrow(cv::Mat srcImage, cv::Point point, int ch1Thres, int ch1LowerBind = 0,int ch1UpperBind = 255)
//{
//	cv::Point growPoint;
//	cv::Mat growImage = cv::Mat::zeros(srcImage.size(), CV_8UC1);
//	std::vector<cv::Point> growPoints;
//	growPoints.push_back(point);
//	growImage.at<uchar>(point.y, point.x) = 255;
//
//	int DIR[8][2] = { (-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1)};
//
//
//	while (!growPoints.empty())
//	{
//		growPoint = growPoints.back();
//		growPoints.pop_back();
//
//		for (int i = 0;i<9;++i)
//		{
//			cv::Point neibourPoint;
//			neibourPoint.x = growPoint.x + DIR[i][0];
//			neibourPoint.y = growPoint.y + DIR[i][1];
//
//			if (neibourPoint.x<0||neibourPoint.y<0||neibourPoint.x>(srcImage.cols-1)||neibourPoint.y>(srcImage.rows-1))
//			{
//				continue;
//			}
//
//			cv::Scalar currentValue = srcImage.at<uchar>(growPoint.y, growPoint.x);
//			cv::Scalar neibourValue = srcImage.at<uchar>(neibourPoint.y, neibourPoint.x);
//			int growValue = growImage.at<uchar>(neibourPoint.y, neibourPoint.x);
//
//			if (growValue==0)
//			{
//				if (neibourValue[0]<=ch1UpperBind&&neibourValue[0]>=ch1LowerBind)
//				{
//					if (abs(neibourValue[0]-currentValue[0])<ch1Thres)
//					{
//						growPoints.push_back(neibourPoint);
//						growImage.at<uchar>(neibourPoint.y, neibourPoint.x) = 255;
//					}
//				}
//			}
//		}
//		
//
//	}
//
//	return growImage.clone();
//
//}
//
//int main()
//{
//	std::string namepath = "D:\\Data\\1305031102.411258.png";
//	cv::Mat img = cv::imread(namepath, cv::IMREAD_GRAYSCALE);
//
//	cv::imshow("img", img);
//	cv::waitKey(0);
//	cv::Point point(480, 175);
//	cv::Mat growImage;
//	growImage = RegionGrow(img, point, 10);
//	cv::imshow("growimage", growImage);
//	cv::waitKey(0);
//
//
//}


//8 手写BA 模拟曲线  //

//using namespace std;
//using namespace Eigen;

//int main()
//{
//	double a = 1.0, b = 2.0, c = 1.0;
//	int N = 100;
//	double w_sigma = 1.0;
//	cv::RNG rng;
//	double ae = 2.0, be = -1.0, ce = 5.0;
//
//	std::vector<double> x_data,y_data;
//	for (int i = 0;i<N;i++)
//	{
//		double x = rng.uniform(0.0, 1.0);
//		//double x = i / 100.0;
//		double y = exp(a*x*x + b*x + c)+rng.gaussian(w_sigma);
//
//		x_data.push_back(x);
//		y_data.push_back(y);
//	}
//	double cost = 0, lastcost = 0;
//
//	int Iterations = 100;
//	for (int iter = 0 ;iter<Iterations;iter++)
//	{
//		Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
//		Eigen::Vector3d b = Eigen::Vector3d::Zero();
//		cost = 0;
//
//		for (int j = 0 ;j<N;j++)
//		{
//			double xi = x_data[j];
//			double yi = y_data[j];
//
//			double error = 0; 
//			error = yi - exp(ae*xi*xi + be*xi + ce);
//
//			Eigen::Vector3d J;
//			J[0] = -xi*xi*exp(ae*xi*xi + be*xi + ce);
//			J[1] = -xi*exp(ae*xi*xi + be*xi + ce);
//			J[2] = -exp(ae*xi*xi + be*xi + ce);
//
//			H += J * J.transpose();
//			b += -error * J;
//
//			cost += error*error;
//		}
//
//		Eigen::Vector3d delta;
//		//delta = H.inverse()*b;
//		delta = H.colPivHouseholderQr().solve(b);
//		if (isnan(delta[0])) {
//			cout << "result is nan!" << endl;
//			break;
//		}
//
//		if (iter> 0 && cost>lastcost)   
//		{
//			break;
//		}
//
//		ae += delta[0];
//		be += delta[1];
//		ce += delta[2];
//
//		std::cout << "cost - lastcost: " << cost - lastcost << std::endl;
//
//		lastcost = cost;
//
//		std::cout << "cost: " << cost << std::endl;
//		
//	}
//
//	std::cout << "a b c: " << ae <<" "<< be << " " << ce;
//
//}

//int main()
//{
//	double a = 1.0, b = 2.0, c = 1.0;
//	double ae = -2.0, be = 1.0, ce = 5.0;
//
//	int N = 100;
//	int Interation = 100;
//
//	double w_sigma = 1.0;
//
//	cv::RNG rng;
//	std::vector<double>x_data, y_data;
//	for (int i =0 ; i<N;i++)
//	{
//		double x = i / 100.0;
//		double y = exp(a*x*x + b*x + c) + rng.gaussian(w_sigma);
//		x_data.push_back(x);
//		y_data.push_back(y);
//	}
//
//	double cost = 0.0,last_cost = 0.0;
//	for (int iter = 0 ;iter<Interation;iter++)
//	{
//		Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
//		Eigen::Vector3d b = Eigen::Vector3d::Zero();
//
//		for (int i = 0 ; i <N;i++)
//		{
//			double xi = x_data[i];
//			double yi = y_data[i];
//			double error = yi - exp(ae*xi*xi + be*xi + ce);
//
//			Eigen::Vector3d J;
//			J[0] = -xi*xi*exp(ae*xi*xi + be*xi + ce);
//			J[1] = -xi*exp(ae*xi*xi + be*xi + ce);
//			J[2] = -exp(ae*xi*xi + be*xi + ce);
//
//			H += J*J.transpose();
//			b += -J*error;
//
//			cost += error;
//		}
//
//		Eigen::Vector3d delta;
//		delta = H.colPivHouseholderQr().solve(b);
//
//		if (isnan(delta[0]))
//		{
//			std::cout << "result is nan" << std::endl;
//		}
//
//		if (iter != 0 && cost > last_cost)
//		{
//			break;
//			std::cout << "cost: " << cost << " last_cost: " << last_cost << std::endl;
//
//		}
//
//		ae += delta[0];
//		be += delta[1];
//		ce += delta[2];
//
//		last_cost = cost;
//	}
//
//	std::cout << "a b c = " << ae << " " << be << " " << ce << std::endl;
//
//	system("pause");
//}

//9 手写BA 求解PNP  自己加了 联合优化点坐标 的部分  //

//只优化帧位姿

//void main()
//{
//	cv::Mat img1 = cv::imread("D:\\Data\\1.png", CV_LOAD_IMAGE_COLOR);
//	cv::Mat img2 = cv::imread("D:\\Data\\2.png", CV_LOAD_IMAGE_COLOR);
//
//	double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
//
//	VecVector2d p2d;
//	VecVector3d p3d;
//
//	detectMatches(img1, img2, p2d, p3d);
//
//	int iterations = 100;
//	double cost = 0;
//	double last_cost = 0;
//	int npoints = p2d.size();
//
//	Sophus::SE3 T_esti;
//
//	for (int iter = 0 ; iter<iterations;iter++)
//	{
//		Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
//		Vector6d b = Vector6d::Zero();
//		cost = 0;
//
//		for (int i = 0; i < npoints; i++)
//		{
//			Vector2d point2D = p2d[i];
//			Vector3d point3D = p3d[i];
//
//			Vector3d p3D_esti = T_esti * point3D;
//			double X = p3D_esti[0];
//			double Y = p3D_esti[1];
//			double Z = p3D_esti[2];
//
//			Vector2d p2D_esti = { fx*(X / Z) + cx,fy*(Y / Z) + cy };
//
//			Vector2d e = point2D  - p2D_esti  ;
//
//			cost += (e[0] * e[0] + e[1] * e[1]);
//
//			Eigen::Matrix<double, 2, 6> J;
//			J(0, 0) = -(fx / Z);
//			J(0, 1) = 0;
//			J(0, 2) = (fx*X / (Z*Z));
//			J(0, 3) = (fx*X*Y / (Z*Z));
//			J(0, 4) = -(fx + (fx*X*X) / (Z*Z));
//			J(0, 5) = fx*Y / Z;
//			J(1, 0) = 0;
//			J(1, 1) = -fy / Z;
//			J(1, 2) = fy*Y / (Z*Z);
//			J(1, 3) = fy + (fy*Y*Y) / (Z*Z);
//			J(1, 4) = -fy*X*Y / (Z*Z);
//			J(1, 5) = -fy*X / Z;
//
//			H += J.transpose()*J;
//			b += -J.transpose()*e;
//
//
//		}
//
//		Vector6d delta = H.ldlt().solve(b);
//
//		if (isnan(delta[0]))
//		{
//			std::cout << "result is nan!" << std::endl;
//		}
//
//	/*	if (iter>0&&cost>=last_cost)
//		{
//			std::cout << "cost " << cost << "last_cost: " << last_cost << std::endl;
//			break;
//		}*/
//
//		if (abs(cost-last_cost)<0.001)
//		{
//			std::cout << "cost " << cost << "last_cost: " << last_cost << std::endl;
//			break;
//		}
//
//		T_esti = Sophus::SE3::exp(delta) * T_esti;
//
//		last_cost = cost;
//		std::cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
//	}
//
//	std::cout << "Estimate pose: " <<std::endl<< T_esti.matrix() << std::endl;
//	Vector6d pose;
//	pose = T_esti.log();
//
//	std::cout << "Eitimate Pose vector "<<std::endl << pose.transpose() << std::endl;
//	system("pause");
//}//

//联合优化帧位姿和点坐标

//void main()
//{
//	cv::Mat img1 = cv::imread("D:\\Data\\1.png", CV_LOAD_IMAGE_COLOR);
//	cv::Mat img2 = cv::imread("D:\\Data\\2.png", CV_LOAD_IMAGE_COLOR);
//
//	VecVector2d p2d;
//	VecVector3d p3d, p3dclone;
//
//	double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
//	detectMatches(img1, img2, p2d, p3d);
//
//	p3dclone = p3d;
//
//	int iterations = 100;
//	double cost = 0;
//	double last_cost = 0;
//	int npoints = p2d.size();
//
//	Sophus::SE3 T_esti;
//	Eigen::Matrix4d T;
//	Eigen::Matrix3d R;
//	
//	for (int iter = 0; iter<iterations; iter++)
//	{
//		Eigen::Matrix<double, 9, 9> H = Eigen::Matrix<double, 9, 9>::Zero();
//		Vector9d b = Vector9d::Zero();
//		cost = 0;
//
//		T = T_esti.matrix();
//		R = T.block<3, 3>(0, 0);
//
//		for (int i = 0; i < npoints; i++)
//		{
//			Vector2d point2D = p2d[i];
//			Vector3d point3D = p3d[i];
//
//			Vector3d p3D_esti = T_esti * point3D;
//			double X = p3D_esti[0];
//			double Y = p3D_esti[1];
//			double Z = p3D_esti[2];
//
//			Vector2d p2D_esti = { fx*(X / Z) + cx,fy*(Y / Z) + cy };
//
//			Vector2d e = point2D - p2D_esti;
//
//			cost += (e[0] * e[0] + e[1] * e[1]);
//
//			Eigen::Matrix<double, 2, 9> J;
//			Eigen::Matrix<double, 2, 3> P;
//
//			J(0, 0) = -(fx / Z);
//			J(0, 1) = 0;
//			J(0, 2) = (fx*X / (Z*Z));
//			J(0, 3) = (fx*X*Y / (Z*Z));
//			J(0, 4) = -(fx + (fx*X*X) / (Z*Z));
//			J(0, 5) = fx*Y / Z;
//			J(1, 0) = 0;
//			J(1, 1) = -fy / Z;
//			J(1, 2) = fy*Y / (Z*Z);
//			J(1, 3) = fy + (fy*Y*Y) / (Z*Z);
//			J(1, 4) = -fy*X*Y / (Z*Z);
//			J(1, 5) = -fy*X / Z;
//
//			P(0, 0) = -fx / Z;
//			P(0, 1) = 0;
//			P(0, 2) = fx*X / (Z*Z);
//			P(1, 0) = 0;
//			P(1, 1) = -fy / Z;
//			P(1, 2) = fy*Y / (Z*Z);
//
//			P = P*R;
//
//			J(0, 6) = P(0, 0);
//			J(0, 7) = P(0, 1);
//			J(0, 8) = P(0, 2);
//
//			J(1, 6) = P(1, 0);
//			J(1, 7) = P(1, 1);
//			J(1, 8) = P(1, 2);
//
//			H += J.transpose()*J;
//			b += -J.transpose()*e;
//
//
//		}
//
//		Vector9d delta = H.ldlt().solve(b);
//		Vector6d delta_pose = delta.block<6, 1>(0, 0);
//		Vector3d delta_point = delta.block<3, 1>(6, 0);
//
//		if (isnan(delta[0]))
//		{
//			std::cout << "result is nan!" << std::endl;
//		}
//
//		/*	if (iter>0&&cost>=last_cost)
//		{
//		std::cout << "cost " << cost << "last_cost: " << last_cost << std::endl;
//		break;
//		}*/
//
//		if (abs(cost - last_cost)<0.001)
//		{
//			std::cout << "cost " << cost << "last_cost: " << last_cost << std::endl;
//			break;
//		}
//
//		T_esti = Sophus::SE3::exp(delta_pose) * T_esti;
//		for (int j = 0;j<npoints;j++)
//		{
//			p3d[j] += delta_point;
//		}
//
//		last_cost = cost;
//		std::cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
//	}
//
//	std::cout << "Estimate pose: " << std::endl << T_esti.matrix() << std::endl;
//	Vector6d pose;
//	pose = T_esti.log();
//
//	std::cout << "优化前后3D点坐标变化：" << std::endl;
//	for (int i = 0; i < npoints; i++)
//	{
//		std::cout << "ID: " << i << std::endl;
//		std::cout << p3dclone[i].transpose() << std::endl;
//		std::cout << p3d[i].transpose() << std::endl;
//	}
//	std::cout << "Eitimate Pose vector " << std::endl << pose.transpose() << std::endl;
//
//
//}

//10 g2o实现pnp  //

//void main()
//{
//	cv::Mat img1 = cv::imread("D:\\Data\\1.png", CV_LOAD_IMAGE_COLOR);
//	cv::Mat img2 = cv::imread("D:\\Data\\2.png", CV_LOAD_IMAGE_COLOR);
//
//	double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
//	VecVector2d p2d1,p2d2;
//	VecVector3d p3d, p3dclone;
//
//	findCorrespondPoints(img1, img2, p2d1, p2d2, p3d);
//
//	p3dclone = p3d;
//
//	g2o::SparseOptimizer optimizer;
//	g2o::BlockSolver_6_3::LinearSolverType *linearsolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
//	g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearsolver);
//	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	
//	optimizer.setAlgorithm(solver);
//	optimizer.setVerbose(true);
//
//	for (int i  =0 ; i <2;i++)
//	{
//		g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
//		v->setId(i);
//		if (i == 0)
//		{
//			v->setFixed(true);
//		}
//		v->setEstimate(g2o::SE3Quat());
//		optimizer.addVertex(v);
//	}
//
//	for (int i =0 ; i<p3d.size();i++)
//	{
//		Eigen::Vector3d point = p3d[i];
//		g2o::VertexSBAPointXYZ *p = new g2o::VertexSBAPointXYZ();
//		p->setId(i+2);
//		p->setMarginalized(true);
//		p->setEstimate(point);
//		optimizer.addVertex(p);
//	}
//	
//	g2o::CameraParameters *camera = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
//	camera->setId(0);
//	optimizer.addParameter(camera);
//
//	std::vector<g2o::EdgeProjectXYZ2UV*> edges;
//	for (int i =0; i <p2d1.size();i++)
//	{
//		g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
//		Eigen::Vector2d p = p2d1[i];
//		edge->setVertex(0, static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + 2)));
//		edge->setVertex(1, static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)));
//		edge->setMeasurement(p);
//		edge->setInformation(Eigen::Matrix2d::Identity());
//		edge->setParameterId(0, 0);
//		edge->setRobustKernel(new g2o::RobustKernelHuber());
//
//		optimizer.addEdge(edge);
//		edges.push_back(edge);
//	}
//
//	for (int i = 0; i < p2d2.size(); i++)
//	{
//		g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
//		Eigen::Vector2d p = p2d2[i];
//		edge->setVertex(0, static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + 2)));
//		edge->setVertex(1, static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)));
//		edge->setMeasurement(p);
//		edge->setInformation(Eigen::Matrix2d::Identity());
//		edge->setParameterId(0, 0);
//		edge->setRobustKernel(new g2o::RobustKernelHuber());
//
//		optimizer.addEdge(edge);
//		edges.push_back(edge);
//	}
//	std::cout << "开始优化： " << std::endl;
//	optimizer.setVerbose(true);
//	optimizer.initializeOptimization();
//	optimizer.optimize(10);
//	std::cout << "优化完毕" << std::endl;
//
//	g2o::VertexSE3Expmap *SE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
//	g2o::SE3Quat SE3Quat_recov = SE3_recov->estimate();
//	cv::Mat poseMat = toCVMat(SE3Quat_recov);
//	Eigen::Isometry3d Pose = SE3_recov->estimate();
//	Vector6d poseVetor = SE3Quat_recov.log();
//
//	std::cout << "g2O Pose: \n" << Pose.matrix() << std::endl;
//	std::cout << "g2O Pose Vector : \n" <<poseVetor.transpose() << std::endl;
//
//	for (int i = 0 ; i<p3d.size();i++)
//	{
//		Vector3d p = p3d[i];
//		g2o::VertexSBAPointXYZ *v = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+2));
//		std::cout << "Vertex id: " << i + 2 << std::endl;
//		Eigen::Vector3d pos = v->estimate();
//		std::cout << "前：" << p[0] << " " << p[1] << " "<< p[2] << std::endl;
//		std::cout << "后：" << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
//	}
//
//	int inliers = 0;
//	for (auto e:edges)
//	{
//		e->computeError();
//		if (e->chi2()>1)
//		{
//			std::cout << "error: " << e->chi2() << std::endl;
//		}
//		else
//		{
//			inliers++;
//		}
//	}
//	std::cout << "inliers in total points: " << inliers << "/ "<<p2d1.size()+p2d2.size()<< std::endl;
//
//	system("pause");
//}

//11 友元& 指针vector测试

//class mypoint
//{
//	friend class test;
//
//public:
//	mypoint(int v) : _value(v) {};
//	~mypoint() {};
//
//	int getvalue() { return _value; }
//	void setvalue(int v) { _value = v; }
//
//private:
//	int _value;
//};
//
//class test
//{
//public:
//	test();
//	~test();
//	void setmypointValue(int x, mypoint *a)
//	{
//		a->_value = 5;
//	}
//
//private:
//
//};
//
//test::test()
//{
//}
//
//test::~test()
//{
//}
//
//class Frame
//{
//public:
//	Frame() :_point(nullptr) {};
//	~Frame();
//
//	
//	int addvalue() 
//	{
//		int a = _point->getvalue();
//		_point->setvalue(a + 100);
//		std::cout << "a: " << std::endl;
//		return a;
//	}
//
//	mypoint *_point;
//
//};
//
//
//Frame::~Frame()
//{
//}
//
//
//void main()
//{
//	std::vector<mypoint*> points;
//	mypoint* p1 = new mypoint(0);
//	mypoint* p2 = new mypoint(0); 
//	mypoint* p3 = new mypoint(0);
//
//	mypoint p4(0);
//	p1->setvalue(1);
//	p2->setvalue(2);
//	p3->setvalue(3);
//
//	points.push_back(p1);
//	points.push_back(p2);
//	points.push_back(p3);
//
//	std::vector<mypoint*> points_local;
//	points_local.push_back(p2);
//	points_local.push_back(p3);
//
//	std::vector<mypoint*> &mypoints_clone = points_local;
//
//	Frame *frame = new Frame();
//	frame->_point = p1;
//	frame->addvalue();
//
//
//	for (std::vector<mypoint*>::const_iterator it = mypoints_clone.begin();it!= points_local.end();it++)
//	{
//		mypoint *myp = *it;
//		myp->setvalue(8);
//	}
//
//	std::cout << "end" << std::endl;
//}


//12 线程互斥量测试 

//std::mutex mutex_total;
//std::mutex mutex_a;
//std::mutex mutex_b;
//
//int a = 0;
//int b = 0;
//int totalNum = 100;
//void test_thread1()
//{
//	while (totalNum > 0)
//	{
//		std::unique_lock<std::mutex> lock_total(mutex_total);
//
//		totalNum--;
//		cout << "thread1 totalNum " << totalNum << endl;
//
//		//lock_total.unlock();
//
//		/*std::unique_lock<std::mutex> lock_a(mutex_total);
//		a++;
//		cout << "thread1 a " << a << endl;
//		lock_a.unlock();
//
//
//		b++;
//		cout << "thread1 b " << b << endl;
//		*/
//
//		Sleep(100);
//
//		
//	}
//
//}
//
//void test_thread2()
//{
//	while (totalNum > 0)
//	{
//		std::unique_lock<std::mutex> lock_total(mutex_total);
//
//		totalNum--;
//		cout << "thread2 totalNum " << totalNum << endl;
//
//		//lock_total.unlock();
//
//		/*std::unique_lock<std::mutex> lock_a(mutex_total);
//		a++;
//		cout << "thread2 a " << a << endl;
//		lock_a.unlock();
//
//		b++;
//		cout << "thread2 b " << b << endl;
//		Sleep(100);*/
//		
//	}
//}
//
//
//void main()
//{
//	std::thread task1(test_thread1);
//	std::thread task2(test_thread2);
//	task1.detach();
//	task2.detach();
//
//	system("pause");
//}

//13 杨辉三角

//void main()
//{
//	int h;
//	std::cout << "输入层数： " << std::endl;
//	std::cin >> h;
//	std::vector<std::vector<int>> yanghuiT;
//	yanghuiT.resize(h);
//
//	yanghuiT[0].push_back(1);
//	yanghuiT[1].push_back(1);
//	yanghuiT[1].push_back(1);
//
//	for (int i = 2 ; i < h;i++)
//	{
//		yanghuiT[i].push_back(1);
//
//		for (int j = 0; j<i-1;j++)
//		{
//			yanghuiT[i].push_back(yanghuiT[i - 1][j] + yanghuiT[i - 1][j + 1]);
//		}
//		yanghuiT[i].push_back(1);
//	}
//
//	for (int i = 0; i < yanghuiT.size(); i++)
//	{
//		for (int j = 0 ; j <yanghuiT[i].size();j++)
//		{
//			std::cout << yanghuiT[i][j] << " ";
//		}
//		std::cout << std::endl;
//	}
//	
//	system("pause");
//}

//14 给定n*n 数组 逆时针旋转 时间复杂度为O(1)

//void anticlock(vector<vector<int>> &A)
//{
//	vector<vector<int>> B(A);
//	for (int i = 0 ; i < A.size();i++)
//	{
//		for (int j = 0; j< A[0].size();j++)
//		{
//			B[i][j] = A[j][A.size()-i-1];
//			std::cout << B[i][j] << " ";
//			
//		}
//		std::cout << std::endl;
//	}
//}
//int main()
//{
//	std::cout << "输入数组维数：" << std::endl;
//	int n;
//	std::cin >> n;
//	std::vector<std::vector<int>> A(n,vector<int>(n,0));
//	cv::RNG rng;
//	for (int i = 0; i<A.size();i++)
//	{
//		for (int j = 0 ; j <A[i].size();j++)
//		{
//			A[i][j] = rng.uniform(0, 10);
//			std::cout << A[i][j] << " ";
//		}
//		std::cout<<std::endl;
//	}
//
//	std::cout << "逆时针旋转90度后： " << std::endl;
//
//	anticlock(A);
//	
//
//	system("pause");
//}


//15 opencv获取鼠标左键位置 联合连通域

//cv::Mat RegionGrow(cv::Mat srcImage, cv::Point point, int ch1Thres, int ch1LowerBind = 0,int ch1UpperBind = 255)
//{
//	cv::Point growPoint;
//	cv::Mat growImage = cv::Mat::zeros(srcImage.size(), CV_8UC1);
//	std::vector<cv::Point> growPoints;
//	growPoints.push_back(point);
//	growImage.at<uchar>(point.y, point.x) = 255;
//
//	int DIR[8][2] = { (-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1)};
//
//
//	while (!growPoints.empty())
//	{
//		growPoint = growPoints.back();
//		growPoints.pop_back();
//
//		for (int i = 0;i<9;++i)
//		{
//			cv::Point neibourPoint;
//			neibourPoint.x = growPoint.x + DIR[i][0];
//			neibourPoint.y = growPoint.y + DIR[i][1];
//
//			if (neibourPoint.x<0||neibourPoint.y<0||neibourPoint.x>(srcImage.cols-1)||neibourPoint.y>(srcImage.rows-1))
//			{
//				continue;
//			}
//
//			cv::Scalar currentValue = srcImage.at<uchar>(growPoint.y, growPoint.x);
//			cv::Scalar neibourValue = srcImage.at<uchar>(neibourPoint.y, neibourPoint.x);
//			int growValue = growImage.at<uchar>(neibourPoint.y, neibourPoint.x);
//
//			if (growValue==0)
//			{
//				if (neibourValue[0]<=ch1UpperBind&&neibourValue[0]>=ch1LowerBind)
//				{
//					if (abs(neibourValue[0]-currentValue[0])<ch1Thres)
//					{
//						growPoints.push_back(neibourPoint);
//						growImage.at<uchar>(neibourPoint.y, neibourPoint.x) = 255;
//					}
//				}
//			}
//		}
//		
//
//	}
//
//	return growImage.clone();
//
//}
//
//cv::Point GrowPoint;
//void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
//{
//	cv::Mat *im = reinterpret_cast<cv::Mat*>(userdata);
//	cv::Point p(x, y);
//
//	switch (EVENT)
//	{
//	case CV_EVENT_LBUTTONDOWN:
//	{
//		GrowPoint = p;
//		//cv::circle(*im, p, 2, cv::Scalar(255), 3);
//	}
//	break;
//
//	}
//
//}
//
//int main()
//{
//	std::string namepath = "D:\\Data\\1305031102.411258.png";
//	cv::Mat img = cv::imread(namepath, cv::IMREAD_GRAYSCALE);
//
//	cv::imshow("img", img);
//	cv::setMouseCallback("img", on_mouse, reinterpret_cast<void*> (&img));
//	while (1)
//	{
//		cv::Mat growImage;
//		growImage = RegionGrow(img, GrowPoint, 5);
//		
//		cv::imshow("growimage", growImage);
//		cv::waitKey(100);
//	}
//
//
//}

//16 标记出二值图像中的连通域个数和每个连通域对应的面积 

//typedef struct FeatureArea
//{
//	int lable;
//	int area;
//	cv::Rect rectangle;
//
//}featureArea;
//
//
//std::vector<FeatureArea> regionGrow(cv::Mat& src, cv::Mat& dst,int minThresold,int maxThresold)
//{
//	std::vector<FeatureArea> Areas;
//	cv::Mat growImg = cv::Mat::zeros(src.size(),CV_8UC1);
//
//	//int neighbour[4][2] = { (0,1),(-1,0),(0,-1),(1,0)};
//	std::vector<cv::Point> neighbour;
//	neighbour.emplace_back(0, 1);
//	neighbour.emplace_back(-1, 0);
//	neighbour.emplace_back(0, -1);
//	neighbour.emplace_back(1, 0);
//
//
//	int lable = 10;
//
//	for (int r = 0 ; r < src.rows;r++)
//	{
//		for (int c = 0 ; c<src.cols;c++)
//		{
//			std::vector<cv::Point> growPoints;
//
//			if (src.at<uchar>(r,c) == 255 && growImg.at<uchar>(r,c) == 0)
//			{
//
//				FeatureArea Area;
//				int area = 1;
//				cv::Point growpoint(r, c);
//				growPoints.push_back(growpoint);
//
//				int min_nx = growpoint.x;
//				int max_nx = growpoint.x;
//				int min_ny = growpoint.y;
//				int max_ny = growpoint.y;
//
//
//				growImg.at<uchar>(r, c) = lable;
//
//				while (!growPoints.empty())
//				{
//					cv::Point p = growPoints.back();
//					growPoints.pop_back();
//					
//
//					for (int i =0 ; i<4; i++)
//					{
//
//						int nx = p.x + neighbour[i].x;
//						int ny = p.y + neighbour[i].y;
//
//						if (nx < 0 || nx > (src.rows-1) || ny < 0 || ny > (src.cols-1))
//						{
//							continue;
//						}
//
//						if (src.at<uchar>(nx, ny) == 255 && growImg.at<uchar>(nx,ny) == 0)
//						{
//							area+=1;
//
//							if (nx>max_nx)
//							{
//								max_nx = nx;
//							}
//							if (nx<min_nx)
//							{
//								min_nx = nx;
//							}
//							if (ny>max_ny)
//							{
//								max_ny = ny;
//							}
//							if (ny<min_ny)
//							{
//								min_ny = ny;
//							}
//							growPoints.push_back(cv::Point(nx, ny));
//							growImg.at<uchar>(nx, ny) = lable;
//						}
//					
//					}
//				}
//
//				if (area > 2)
//				{
//					Area.area = area;
//					Area.lable = lable;
//					cv::Rect box = cv::Rect(min_ny, min_nx, max_ny - min_ny + 1, max_nx - min_nx + 1);
//					Area.rectangle = box;
//
//					Areas.push_back(Area);
//
//					lable ++;
//				}
//
//			}
//		}
//	}
//
//
//
//	dst = growImg.clone();
//	return Areas;
//}
//void main()
//{
//	cv::Mat img = cv::imread("D:\\Data\\1.jpg",0);
//	cv::threshold(img, img, 150, 255, CV_THRESH_BINARY);
//
//	cv::imshow("原二值化图像： ", img);
//	cv::waitKey(100);
//
//	std::vector<featureArea> Areas;
//	cv::Mat growImage;
//
//	Areas = regionGrow(img, growImage, 0, 255);
//
//
//	
//	for (int i =0 ;  i <growImage.rows;i++)
//	{
//		for (int j = 0; j < growImage.cols; j++)
//		{
//			growImage.at<uchar>(i, j) = growImage.at<uchar>(i, j) * 30;
//		}
//	}
//
//	std::cout << "共有连通域： " << Areas.size() << std::endl;
//	std::cout << "标号"<< "\t" << "面积"<<std::endl;
//
//	for (int i = 0; i < Areas.size(); i++)
//	{
//		std::cout << i << "\t" << Areas[i].area << std::endl;
//		cv::rectangle(growImage, Areas[i].rectangle, 255);
//	}
//
//	cv::imshow("连通域图像: ", growImage);
//	cv::waitKey(0);
//
//}

//17例 string操作

//int main()
//{
//	/*string S = "123456";
//	string subS;
//	string::iterator it = subS.begin();
//	string::iterator itrS = std::find(S.begin(),S.end(), '4');
//
//	subS.insert(it, S.begin(), itrS);
//	std::cout << subS << std::endl;*/
//
//
//	vector<int> A{ 1,2,3,4,5 };
//	//auto minmax = minmax_element(A.begin(),A.end());
//	//int minA = *minmax.first;
//	//int maxA = *minmax.second;
//
//	//auto minmaxA = std::minmax(A.begin(),A.end());
//	//int minA = *minmaxA.first;
//	//int maxA = *minmaxA.second;
//
//	int minA = *std::min_element(A.begin(), A.end());
//	int maxA = *std::max_element(A.begin(), A.end());
//
//	std::cout << minA << maxA << std::endl;
//
//	int count = 0;
//	unsigned int flag = 1;
//	while (flag)
//	{
//		if (3 & flag)
//			count++;
//
//		flag = flag << 1;
//		std::cout << count << std::endl;
//	}
//
//	system("pause");
//	return count;
//
//}

//18 opencv前景提取

//int main()
//{
//	cv::VideoCapture capture("D:\\768X576.avi");
//	if (!capture.isOpened())
//	{
//		return 0;
//	}
//
//	cv::Mat Frame, FrameGray;
//	cv::Mat foreground;
//	cv::namedWindow("Extracted ForeGround");
//	cv::BackgroundSubtractorMOG mog;
//	bool stop(false);
//
//
//
//
//	while (!stop)
//	{
//		if (!capture.read(Frame))
//		{
//			break;
//		}
//
//		cv::cvtColor(Frame, FrameGray, CV_BGR2GRAY);
//		mog(FrameGray, foreground, 0.01);
//
//		cv::imshow("Extracted ForeGround", foreground);
//		cv::imshow("image", Frame);
//
//		if (cv::waitKey(10) >= 0)
//		{
//			stop = true;
//		}
//
//	}
//
//	return 0;
//}

//19 轨迹对比

//void saveTrajectory(std::vector<cv::Point3f> points, string& outputname)
//{
//	std::ofstream file(outputname, std::ios::trunc | std::ios::in);
//	for (cv::Point3f p : points)
//	{
//		file << setiosflags(ios::fixed) <<std::setprecision(6)
//			 << p.x << " " << p.y << " " << p.z << "\n";
//	}
//
//}
//int main()
//{
//	string path1 = "C:\\Users\\Administrator\\Desktop\\CameraTrajectory.txt";
//	string path2 = "C:\\Users\\Administrator\\Desktop\\groundtruth.txt";
//
//	std::ifstream file;
//	file.open(path1);
//	if (!file.is_open())
//	{
//		std::cout << "Load file error!" << std::endl;
//	}
//	std::vector<cv::Point3f> pointset;
//	string line;
//	while (!file.eof())
//	{
//		cv::Point3f point;
//		getline(file, line);
//		if (!line.empty())
//		{
//			int timestamp;
//			stringstream in;
//			in << line;
//			in >> timestamp>> point.x >> point.y >> point.z;
//			pointset.push_back(point);
//		}
//	}
//	string output_path = "D:\\Data\\trajectory_test.txt";
//	saveTrajectory(pointset, output_path);
//
//
//	std::ifstream file2;
//	file2.open(path2);
//	if (!file2.is_open())
//	{
//		std::cout << "Load file error!" << std::endl;
//	}
//	std::vector<cv::Point3f> pointset2;
//	string line2;
//
//	getline(file2, line2);
//	getline(file2, line2);
//	getline(file2, line2);
//
//	while (!file2.eof())
//	{
//		cv::Point3f point;
//		getline(file2, line2);
//		if (!line2.empty())
//		{
//			stringstream in;
//			double timestamp;
//			in << line2;
//			in >> timestamp >>point.x >> point.y >> point.z;
//			pointset2.push_back(point);
//		}
//
//	}
//
//	string output_path2 = "D:\\Data\\trajectory_groundtruth.txt";
//	saveTrajectory(pointset2, output_path2);
//
//	system("pause");
//
//
//}


// 20 椭圆拟合

//void fitEllipse2f(const std::vector<cv::Point>& contours, cv::Mat& params)
//{
//	//long double x1 = 0;
//	//long double x2 = 0;
//	//long double x3 = 0;
//	////long double x4 = 0;
//	//long double y1 = 0;
//	//long double y2 = 0;
//	//long double y3 = 0;
//	//long double y4 = 0;
//	//long double x1y1 = 0;
//	//long double x1y2 = 0;
//	//long double x1y3 = 0;
//	//long double x2y1 = 0;
//	//long double x2y2 = 0;
//	//long double x3y1 = 0;
//
//	double x1 = 0;
//	double x2 = 0;
//	double x3 = 0;
//	//doublete x4 = 0;
//	double y1 = 0;
//	double y2 = 0;
//	double y3 = 0;
//	double y4 = 0;
//	double x1y1 = 0;
//	double x1y2 = 0;
//	double x1y3 = 0;
//	double x2y1 = 0;
//	double x2y2 = 0;
//	double x3y1 = 0;
//
//	int num;
//	std::vector<cv::Point>::const_iterator k;
//	num = contours.size();
//	for (k = contours.begin(); k != contours.end(); k++)
//	{
//		x1 = x1 + (*k).x;
//		x2 = x2 + pow((*k).x, 2);
//		x3 = x3 + pow((*k).x, 3);
//		//x4 = x4 + pow((*k).x, 4);
//		y1 = y1 + (*k).y;
//		y2 = y2 + pow((*k).y, 2);
//		y3 = y3 + pow((*k).y, 3);
//		y4 = y4 + pow((*k).y, 4);
//		x1y1 = x1y1 + (*k).x * (*k).y;
//		x1y2 = x1y2 + (*k).x * pow((*k).y, 2);
//		x1y3 = x1y3 + (*k).x * pow((*k).y, 3);
//		x2y1 = x2y1 + pow((*k).x, 2) * (*k).y;
//		x2y2 = x2y2 + pow((*k).x, 2) * pow((*k).y, 2);
//		x3y1 = x3y1 + pow((*k).x, 3) * (*k).y;
//	}
//
//
//	cv::Mat left_matrix = (cv::Mat_<double>(5, 5) << x2y2, x1y3, x2y1, x1y2, x1y1,
//		x1y3, y4, x1y2, y3, y2,
//		x2y1, x1y2, x2, x1y1, x1,
//		x1y2, y3, x1y1, y2, y1,
//		x1y1, y2, x1, y1, num);
//
//
//	cv::Mat right_matrix = (cv::Mat_<double>(5, 1) << -x3y1, -x2y2, -x3, -x2y1, -x2);
//
//	cv::Mat ellipse_solution(5, 1, CV_64F);
//
//	solve(left_matrix, right_matrix, ellipse_solution, cv::DECOMP_LU);
//
//	//ellipse_solution = left_matrix.inv()*right_matrix;
//
//	params = ellipse_solution.clone();
//
//}
//double calfiterror(const std::vector<cv::Point>& contours, cv::Mat& params)
//{
//	double A = params.at<double>(0, 0);
//	double B = params.at<double>(1, 0);
//	double C = params.at<double>(2, 0);
//	double D = params.at<double>(3, 0);
//	double E = params.at<double>(4, 0);
//
//	double error = 0;
//
//	for (int i = 0; i < contours.size(); i++)
//	{
//		cv::Point temp = contours[i];
//		int x = temp.x;
//		int y = temp.y;
//		error += abs(x*x + A*x*y + B*y*y + C*x + D*y + E);
//	}
//	error /= contours.size();
//	return error;
//
//}
//int main()
//{
//	std::vector<cv::Point> Points;
//	string pathname = "D:\\point.txt";
//	std::ifstream file;
//	file.open(pathname.c_str());
//
//	
//	while (!file.eof())
//	{
//		cv::Point p;
//		file >> p.x >> p.y;
//		Points.push_back(p);
//	}
//
//	cv::Mat params;
//	fitEllipse2f(Points, params);
//	//cv::Mat pclone = params;         //Mat内部也是一个指针，这样直接赋值是浅拷贝 一个改变 另一个也会改变 mat.clone()是深拷贝
//	//pclone.at<double>(0, 0) = 0.0;
//
//	double A = params.at<double>(0, 0);
//	double B = params.at<double>(1, 0);
//	double C = params.at<double>(2, 0);
//	double D = params.at<double>(3, 0);
//	double E = params.at<double>(4, 0);
//
//	calfiterror(Points, params);
//
//	std::cout << A << " " << B << " " << C << " " << D << " " << E << std::endl;
//
//
//	std::cout << "end" << std::endl;
//
//}

//21 

//class animal
//{
//public:
//	animal() {}
//
//	animal(int height, int weight):_height(height),_weight(weight)
//	{
//		cout << "animal construct" << endl;
//	}
//
//public:
//	int _height;
//	int _weight;
//
//};
//
//class fish :public animal
//{
//public:
//	fish() :_legs(0)
//	{}
//
//	fish(int leg):animal(400, 300),_legs(leg)
//	{
//		cout << "fish construct" << endl;
//	}
//
//	fish(animal& animals,int leg) :animal(animals), _legs(leg) 
//	{
//		_heights = animals._height;
//	}
//	
//public:
//	int _legs;
//	int _heights;
//};
//
//void main()
//{
//	animal * A = new animal(300, 200);
//
//	fish *fh = new fish(*A, 0);
//
//	A->_height = 800;
//
//	std::cout << fh->_heights << std::endl;
//	system("pause");
//}

//22 Ax=0

//int main()
//{
//	cv::Mat A = (cv::Mat_<double>(3, 4) << 1, 2, 2, 2,
//		2, 4, 6, 8,
//		3, 6, 8, 10);
//
//	/*cv::Mat A = (cv::Mat_<double>(3, 3) << 1, 0, 1,
//		0, 1, 2,
//		0, 0, 0);*/
//	cv::Mat W,U,V;
//	cv::SVD::compute(A, W, U, V);
//	V = V.t();
//	cv::Mat ans = V.col(V.cols-1);
//	cv::Mat vec;
//	cv::SVD::solveZ(A, vec);
//	system("pause");
//}

//23

//int main()
//{
//	std::vector<int> list;
//	list.push_back(2);
//	list.push_back(3);
//
//	int N = list.size();
//	float **array2D = new float *[N];
//	for (int i = 0; i < N; ++i)
//	{
//		array2D[i] = new float[N];
//	}
//
//}

//24 LK光流法

//void main()
//{
//
//	string path_to_dataset = "D:\\Data\\data";
//	string associate_file = path_to_dataset + "\\associate.txt";
//
//	ifstream fin(associate_file);
//	if (!fin)
//	{
//		cerr << "I cann't find associate.txt!" << endl;
//		return;
//	}
//
//	string rgb_file, depth_file, time_rgb, time_depth;
//	list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
//
//	cv::Mat color, depth, last_color;
//
//	for (int index = 0; index < 100; index++)
//	{
//		fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
//		color = cv::imread(path_to_dataset + "\\" + rgb_file);
//		depth = cv::imread(path_to_dataset + "\\" + depth_file, -1);
//
//		// 对第一帧提取FAST特征点
//	
//		if (index == 0)
//		{
//			vector<cv::KeyPoint> kps;
//			kps.reserve(10000);
//			cv::Ptr<cv::OrbFeatureDetector> detector = cv::OrbFeatureDetector::create("ORB");
//			//cv::FastFeatureDetector detector(10);
//			detector->detect(color, kps);
//			for (auto kp : kps)
//				keypoints.push_back(kp.pt);
//			last_color = color.clone();
//	
//			continue;
//		}
//	
//		if (color.empty()||depth.empty())
//		{
//			break;
//		}
//
//		// 对其他帧用LK跟踪特征点
//		vector<cv::Point2f> next_keypoints;
//		vector<cv::Point2f> prev_keypoints;
//		prev_keypoints.reserve(10000);
//		next_keypoints.reserve(10000);
//		for (auto kp : keypoints)
//			prev_keypoints.push_back(kp);
//		vector<unsigned char> status;
//		vector<float> error;
//		status.reserve(10000);
//		error.reserve(10000);
//		cv::calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints, status, error);
//
//		// 把跟丢的点删掉
//		int i = 0;
//		for (auto iter = keypoints.begin(); iter != keypoints.end(); i++)
//		{
//			if (status[i] == 0)
//			{
//				iter = keypoints.erase(iter);
//				continue;
//			}
//			*iter = next_keypoints[i];
//			iter++;
//		}
//		cout << "tracked keypoints: " << keypoints.size() << endl;
//		if (keypoints.size() == 0)
//		{
//			cout << "all keypoints are lost." << endl;
//			break;
//		}
//		// 画出 keypoints
//		cv::Mat img_show = color.clone();
//		for (auto kp : keypoints)
//			cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
//		cv::imshow("corners", img_show);
//		cv::waitKey(100);
//		last_color = color.clone();
//	}
//
//	return;
//}


//#define UNKNOWN_FLOW_THRESH 1e9
//
//void makecolorwheel(vector<Scalar> &colorwheel)
//{
//	int RY = 15;
//	int YG = 6;
//	int GC = 4;
//	int CB = 11;
//	int BM = 13;
//	int MR = 6;
//
//	int i;
//
//	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
//	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
//	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
//	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
//	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
//	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
//}
//
//void motionToColor(Mat flow, Mat &color)
//{
//	if (color.empty())
//		color.create(flow.rows, flow.cols, CV_8UC3);
//
//	static vector<Scalar> colorwheel; //Scalar r,g,b
//	if (colorwheel.empty())
//		makecolorwheel(colorwheel);
//
//	// determine motion range:
//	float maxrad = -1;
//
//	// Find max flow to normalize fx and fy
//	for (int i = 0; i < flow.rows; ++i)
//	{
//		for (int j = 0; j < flow.cols; ++j)
//		{
//			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
//			float fx = flow_at_point[0];
//			float fy = flow_at_point[1];
//			if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))
//				continue;
//			float rad = sqrt(fx * fx + fy * fy);
//			maxrad = maxrad > rad ? maxrad : rad;
//		}
//	}
//
//	for (int i = 0; i < flow.rows; ++i)
//	{
//		for (int j = 0; j < flow.cols; ++j)
//		{
//			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
//			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
//
//			float fx = flow_at_point[0] / maxrad;
//			float fy = flow_at_point[1] / maxrad;
//			if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))
//			{
//				data[0] = data[1] = data[2] = 0;
//				continue;
//			}
//			float rad = sqrt(fx * fx + fy * fy);
//
//			float angle = atan2(-fy, -fx) / CV_PI;
//			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
//			int k0 = (int)fk;
//			int k1 = (k0 + 1) % colorwheel.size();
//			float f = fk - k0;
//			//f = 0; // uncomment to see original color wheel
//
//			for (int b = 0; b < 3; b++)
//			{
//				float col0 = colorwheel[k0][b] / 255.0;
//				float col1 = colorwheel[k1][b] / 255.0;
//				float col = (1 - f) * col0 + f * col1;
//				if (rad <= 1)
//					col = 1 - rad * (1 - col); // increase saturation with radius
//				else
//					col *= .75; // out of range
//				data[2 - b] = (int)(255.0 * col);
//			}
//		}
//	}
//}
//
//int main()
//{
//	cv::VideoCapture cap;
//	//cap.open(0);
//	cap.open("D:\\Data\\768X576.avi");
//
//	if (!cap.isOpened())
//		return -1;
//
//	Mat prevgray, gray, flow, cflow, frame;
//	namedWindow("flow", 1);
//
//	Mat motion2color;
//
//	for (;;)
//	{
//		double t = (double)cvGetTickCount();
//
//		cap >> frame;
//		cvtColor(frame, gray, CV_BGR2GRAY);
//		imshow("original", frame);
//
//		if (prevgray.data)
//		{
//			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//			motionToColor(flow, motion2color);
//			imshow("flow", motion2color);
//		}
//		if (waitKey(10) >= 0)
//			break;
//		std::swap(prevgray, gray);
//
//		t = (double)cvGetTickCount() - t;
//		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
//	}
//	return 0;
//}

//int main()
//{
//	std::vector<int *> A;
//	A = std::vector<int*>(5, static_cast<int*>(NULL));
//	std::fill(A.begin(), A.end(), static_cast<int*>(NULL));
//
//	return 0;
//}

//int main()
//{
//	cv::Mat A = (cv::Mat_<int>(2, 2) << 1, 1, 0, 0);
//	cv::Mat B = (cv::Mat_<int>(2, 2) << 0, 3, 0, 0);
//	cv::Mat C = (cv::Mat_<double>(2, 1) << 1., 2.);
//	cv::Mat ans = A | B;
//	double dist = cv::norm(A, B, cv::NORM_L2);
//	int num = A.total();
//	char c[] = "1231";
//	int n = atoi(c);
//	//C /= norm(C);
//	normalize(C, C);
//	normalize(B, B);
//
//	//std::cout << ans << std::endl;
//	
//	return 1;
//}

//#include <sstream>
//#include <iomanip>
//
//class FrameProcessor;
////帧处理基类
//class FrameProcessor {
//public:
//	virtual void process(Mat &input, Mat &ouput) = 0;
//};
//
////特征跟踪类，继承自帧处理基类
//class FeatureTracker : public FrameProcessor {
//	Mat gray;  //当前灰度图
//	Mat gray_prev;  //之前的灰度图
//	vector<Point2f> points[2];//前后两帧的特征点
//	vector<Point2f> initial;//初始特征点
//	vector<Point2f> features;//检测到的特征
//	int max_count; //要跟踪特征的最大数目
//	double qlevel; //特征检测的指标
//	double minDist;//特征点之间最小容忍距离
//	vector<uchar> status; //特征跟踪状态
//	vector<float> err; //跟踪时的错误
//public:
//	FeatureTracker() :max_count(500), qlevel(0.01), minDist(10.) {}
//	void process(Mat &frame, Mat &output) {
//		//得到灰度图
//		cvtColor(frame, gray, CV_BGR2GRAY);
//		frame.copyTo(output);
//		//特征点太少了，重新检测特征点
//		if (addNewPoint()) {
//			detectFeaturePoint();
//			//插入检测到的特征点
//			points[0].insert(points[0].end(), features.begin(), features.end());
//			initial.insert(initial.end(), features.begin(), features.end());
//		}
//		//第一帧
//		if (gray_prev.empty()) {
//			gray.copyTo(gray_prev);
//		}
//		//根据前后两帧灰度图估计前一帧特征点在当前帧的位置
//		//默认窗口是15*15
//		calcOpticalFlowPyrLK(
//			gray_prev,//前一帧灰度图
//			gray,//当前帧灰度图
//			points[0],//前一帧特征点位置
//			points[1],//当前帧特征点位置
//			status,//特征点被成功跟踪的标志
//			err);//前一帧特征点点小区域和当前特征点小区域间的差，根据差的大小可删除那些运动变化剧烈的点
//		int k = 0;
//		//去除那些未移动的特征点
//		for (int i = 0; i < points[1].size(); i++) {
//			if (acceptTrackedPoint(i)) {
//				initial[k] = initial[i];
//				points[1][k++] = points[1][i];
//			}
//		}
//		points[1].resize(k);
//		initial.resize(k);
//		//标记被跟踪的特征点
//		handleTrackedPoint(frame, output);
//		//为下一帧跟踪初始化特征点集和灰度图像
//		std::swap(points[1], points[0]);
//		cv::swap(gray_prev, gray);
//	}
//
//	void detectFeaturePoint() {
//		goodFeaturesToTrack(gray,//图片
//			features,//输出特征点
//			max_count,//特征点最大数目
//			qlevel,//质量指标
//			minDist);//最小容忍距离
//	}
//	bool addNewPoint() {
//		//若特征点数目少于10，则决定添加特征点
//		return points[0].size() <= 10;
//	}
//
//	//若特征点在前后两帧移动了，则认为该点是目标点，且可被跟踪
//	bool acceptTrackedPoint(int i) {
//		return status[i] &&
//			(abs(points[0][i].x - points[1][i].x) +
//				abs(points[0][i].y - points[1][i].y) > 2);
//	}
//
//	//画特征点
//	void  handleTrackedPoint(Mat &frame, Mat &output) {
//		for (int i = 0; i < points[i].size(); i++) {
//			//当前特征点到初始位置用直线表示
//			line(output, initial[i], points[1][i], Scalar::all(0));
//			//当前位置用圈标出
//			circle(output, points[1][i], 3, Scalar::all(0), (-1));
//		}
//	}
//};
//
//
//class VideoProcessor {
//private:
//	VideoCapture caputure;
//	//写视频流对象
//	VideoWriter writer;
//	//输出文件名
//	string Outputfile;
//
//	int currentIndex;
//	int digits;
//	string extension;
//	FrameProcessor *frameprocessor;
//	//图像处理函数指针
//	void(*process)(Mat &, Mat &);
//	bool callIt;
//	string WindowNameInput;
//	string WindowNameOutput;
//	//延时
//	int delay;
//	long fnumber;
//	//第frameToStop停止
//	long frameToStop;
//	//暂停标志
//	bool stop;
//	//图像序列作为输入视频流
//	vector<string> images;
//	//迭代器
//public:
//	VideoProcessor() : callIt(true), delay(0), fnumber(0), stop(false), digits(0), frameToStop(-1) {}
//	//设置图像处理函数
//	void setFrameProcessor(void(*process)(Mat &, Mat &)) {
//		frameprocessor = 0;
//		this->process = process;
//		CallProcess();
//	}
//	//打开视频
//	bool setInput(string filename) {
//		fnumber = 0;
//		//若已打开，释放重新打开
//		caputure.release();
//		return caputure.open(filename);
//	}
//	//设置输入视频播放窗口
//	void displayInput(string wn) {
//		WindowNameInput = wn;
//		namedWindow(WindowNameInput);
//	}
//	//设置输出视频播放窗口
//	void displayOutput(string wn) {
//		WindowNameOutput = wn;
//		namedWindow(WindowNameOutput);
//	}
//	//销毁窗口
//	void dontDisplay() {
//		destroyWindow(WindowNameInput);
//		destroyWindow(WindowNameOutput);
//		WindowNameInput.clear();
//		WindowNameOutput.clear();
//	}
//
//	//启动
//	void run() {
//		Mat frame;
//		Mat output;
//		if (!isOpened())
//			return;
//		stop = false;
//		while (!isStopped()) {
//			//读取下一帧
//			if (!readNextFrame(frame))
//				break;
//			if (WindowNameInput.length() != 0)
//				imshow(WindowNameInput, frame);
//			//处理该帧
//			if (callIt) {
//				if (process)
//					process(frame, output);
//				else if (frameprocessor)
//					frameprocessor->process(frame, output);
//			}
//			else {
//				output = frame;
//			}
//			if (Outputfile.length()) {
//				cvtColor(output, output, CV_GRAY2BGR);
//				writeNextFrame(output);
//			}
//			if (WindowNameOutput.length() != 0)
//				imshow(WindowNameOutput, output);
//			//按键暂停，继续按键继续
//			if (delay >= 0 && waitKey(delay) >= 0)
//				waitKey(0);
//			//到达指定暂停键，退出
//			if (frameToStop >= 0 && getFrameNumber() == frameToStop)
//				stopIt();
//		}
//	}
//	//暂停键置位
//	void stopIt() {
//		stop = true;
//	}
//	//查询暂停标志位
//	bool isStopped() {
//		return stop;
//	}
//	//返回视频打开标志
//	bool isOpened() {
//		return  caputure.isOpened() || !images.empty();
//	}
//	//设置延时
//	void setDelay(int d) {
//		delay = d;
//	}
//	//读取下一帧
//	bool readNextFrame(Mat &frame) {
//		if (images.size() == 0)
//			return caputure.read(frame);
//		else {
//			if (itImg != images.end()) {
//				frame = imread(*itImg);
//				itImg++;
//				return frame.data ? 1 : 0;
//			}
//			else
//				return false;
//		}
//	}
//
//	void CallProcess() {
//		callIt = true;
//	}
//	void  dontCallProcess() {
//		callIt = false;
//	}
//	//设置停止帧
//	void stopAtFrameNo(long frame) {
//		frameToStop = frame;
//	}
//	// 获得当前帧的位置
//	long getFrameNumber() {
//		long fnumber = static_cast<long>(caputure.get((CV_CAP_PROP_POS_FRAMES)));
//		return fnumber;
//	}
//
//	//获得帧大小
//	Size getFrameSize() {
//		if (images.size() == 0) {
//			// 从视频流获得帧大小
//			int w = static_cast<int>(caputure.get(CV_CAP_PROP_FRAME_WIDTH));
//			int h = static_cast<int>(caputure.get(CV_CAP_PROP_FRAME_HEIGHT));
//			return Size(w, h);
//		}
//		else {
//			//从图像获得帧大小
//			cv::Mat tmp = cv::imread(images[0]);
//			return (tmp.data) ? (tmp.size()) : (Size(0, 0));
//		}
//	}
//
//	//获取帧率
//	double getFrameRate() {
//		return caputure.get(CV_CAP_PROP_FPS);
//	}
//	vector<string>::const_iterator itImg;
//	bool setInput(const vector<string> &imgs) {
//		fnumber = 0;
//		caputure.release();
//		images = imgs;
//		itImg = images.begin();
//		return true;
//	}
//
//	void  setFrameProcessor(FrameProcessor *frameprocessor) {
//		process = 0;
//		this->frameprocessor = frameprocessor;
//		CallProcess();
//	}
//
//	//获得编码类型
//	int getCodec(char codec[4]) {
//		if (images.size() != 0)
//			return -1;
//		union { // 数据结构4-char
//			int value;
//			char code[4];
//		} returned;
//		//获得编码值
//		returned.value = static_cast<int>(
//			caputure.get(CV_CAP_PROP_FOURCC));
//		// get the 4 characters
//		codec[0] = returned.code[0];
//		codec[1] = returned.code[1];
//		codec[2] = returned.code[2];
//		codec[3] = returned.code[3];
//		return returned.value;
//	}
//
//
//	bool setOutput(const string &filename, int codec = 0, double framerate = 0.0, bool isColor = true) {
//		//设置文件名
//		Outputfile = filename;
//		//清空扩展名
//		extension.clear();
//		//设置帧率
//		if (framerate == 0.0) {
//			framerate = getFrameRate();
//		}
//		//获取输入原视频的编码方式
//		char c[4];
//		if (codec == 0) {
//			codec = getCodec(c);
//		}
//		return writer.open(Outputfile,
//			codec,
//			framerate,
//			getFrameSize(),
//			isColor);
//	}
//
//	//输出视频帧到图片fileme+currentIndex.ext,如filename001.jpg
//	bool setOutput(const string &filename,//路径
//		const string &ext,//扩展名
//		int numberOfDigits = 3,//数字位数
//		int startIndex = 0) {//起始索引
//		if (numberOfDigits < 0)
//			return false;
//		Outputfile = filename;
//		extension = ext;
//		digits = numberOfDigits;
//		currentIndex = startIndex;
//		return true;
//	}
//
//	//写下一帧
//	void writeNextFrame(Mat &frame) {
//		//如果扩展名不为空，写到图片文件中
//		if (extension.length()) {
//			stringstream ss;
//			ss << Outputfile << setfill('0') << setw(digits) << currentIndex++ << extension;
//			imwrite(ss.str(), frame);
//		}
//		//反之，写到视频文件中
//		else {
//			writer.write(frame);
//		}
//	}
//
//};
//
////帧处理函数：canny边缘检测
//void canny(cv::Mat& img, cv::Mat& out) {
//	//灰度变换
//	if (img.channels() == 3)
//		cvtColor(img, out, CV_BGR2GRAY);
//	// canny算子求边缘
//	Canny(out, out, 100, 200);
//	//颜色反转，看起来更舒服些
//	threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
//}
//
//
//int main(int argc, char *argv[])
//{
//	VideoProcessor processor;
//	FeatureTracker tracker;
//	//打开输入视频
//	processor.setInput("D:\\Data\\768x576.avi");
//	processor.displayInput("Current Frame");
//	processor.displayOutput("Output Frame");
//	//设置每一帧的延时
//	processor.setDelay(1000. / processor.getFrameRate());
//	//设置帧处理函数，可以任意
//	processor.setFrameProcessor(&tracker);
//	//   processor.setOutput ("./bikeout.avi");
//	//    processor.setOutput ("bikeout",".jpg");
//	processor.run();
//	return 0;
//}


//delunay 三角

//#include <SFML/Graphics.hpp>
//
//#include "vector2.h"
//#include "triangle.h"
//#include "delaunay.h"
//
//float RandomFloat(float a, float b) {
//	const float random = ((float)rand()) / (float)RAND_MAX;
//	const float diff = b - a;
//	const float r = random * diff;
//	return a + r;
//}
//
//int main(int argc, char * argv[])
//{
//	int numberPoints = 50;
//	//if (argc == 1)
//	//{
//	//	numberPoints = (int)roundf(RandomFloat(4, numberPoints));
//	//}
//	//else if (argc > 1)
//	//{
//	//	numberPoints = atoi(argv[1]);
//	//}
//	srand(time(NULL));
//
//	std::cout << "Generating " << numberPoints << " random points" << std::endl;
//
//	std::vector<Vector2<float> > points;
//	for (int i = 0; i < numberPoints; ++i) {
//		points.push_back(Vector2<float>(RandomFloat(0, 800), RandomFloat(0, 600)));
//	}
//
//	clock_t start1 = clock();
//	Delaunay<float> triangulation;
//	const std::vector<Triangle<float> > triangles = triangulation.triangulate(points);
//	clock_t end1 = clock();
//	std::cout << "cost time:" << (end1-start1)<<std::endl;
//	std::cout << triangles.size() << " triangles generated\n";
//	const std::vector<Edge<float> > edges = triangulation.getEdges();
//
//	std::cout << " ========= ";
//
//	std::cout << "\nPoints : " << points.size() << std::endl;
//	for (const auto &p : points)
//		std::cout << p << std::endl;
//
//	std::cout << "\nTriangles : " << triangles.size() << std::endl;
//	for (const auto &t : triangles)
//		std::cout << t << std::endl;
//
//	std::cout << "\nEdges : " << edges.size() << std::endl;
//	for (const auto &e : edges)
//		std::cout << e << std::endl;
//
//	//cv::Mat showimage(1000, 1000, CV_8UC3);
//	//for (int i = 0;i<showimage.rows;i++)
//	//{
//	//	for (int j =0 ;j <showimage.cols;j++)
//	//	{
//	//		Vec3b scalar(255,255,255);
//	//		showimage.ptr<Vec3b>(i)[j] = scalar;
//
//	//		showimage.ptr<Vec3b>(i)[j][0] = 255;
//	//		showimage.ptr<Vec3b>(i)[j][1] = 255;
//	//		showimage.ptr<Vec3b>(i)[j][2] = 255;
//
//	//		/*showimage.ptr<uchar>(i)[j * 3] = 255;
//	//		showimage.ptr<uchar>(i)[j * 3 + 1] = 255;
//	//		showimage.ptr<uchar>(i)[j * 3 + 2] = 255;*/
//	//	}
//	//}
//	//for (const auto &e : edges)
//	//{
//	//	Vector2<float> p1 = e.p1;
//	//	Vector2<float> p2 = e.p2;
//	//	cv::Point2f pbegin(p1.x, p1.y);
//	//	cv::Point2f pend(p2.x, p2.y);
//	//	cv::line(showimage, pbegin, pend, cv::Scalar(0, 255, 0), 1);
//	//}
//
//	//cv::imshow("showimage", showimage);
//	//cv::waitKey(100);
//	//SFML window;
//	sf::RenderWindow window(sf::VideoMode(800, 600), "Delaunay triangulation");
//
//    // Transform each points of each vector as a rectangle
//	std::vector<sf::RectangleShape*> squares;
//
//	for (const auto p : points) {
//		sf::RectangleShape *c1 = new sf::RectangleShape(sf::Vector2f(4, 4));
//		c1->setPosition(p.x, p.y);
//		squares.push_back(c1);
//	}
//
//	//Make the lines
//	std::vector<std::array<sf::Vertex, 2> > lines;
//	for (const auto &e : edges) {
//		lines.push_back({ {
//				sf::Vertex(sf::Vector2f(e.p1.x + 2, e.p1.y + 2)),
//				sf::Vertex(sf::Vector2f(e.p2.x + 2, e.p2.y + 2))
//			} });
//	}
//
//	while (window.isOpen())
//	{
//		sf::Event event;
//		while (window.pollEvent(event))
//		{
//			if (event.type == sf::Event::Closed)
//				window.close();
//		}
//
//		window.clear();
//
//		//Draw the squares
//		for (const auto &s : squares) {
//			window.draw(*s);
//		}
//
//		//Draw the lines
//		for (const auto &l : lines) {
//			window.draw(l.data(), 2, sf::Lines);
//		}
//
//		window.display();
//	}
//
//	return 0;
//}


//int main()
//{
//
//
//	vector<int> L;
//	L.push_back(1);
//	L.push_back(2);
//	L.push_back(3);
//	L.push_back(4);
//	L.push_back(5);
//	L.push_back(3);
//	vector<int>::iterator result = L.begin();
//	while (result !=L.end())
//	{
//		result = std::find(result, L.end(), 3);
//		if (result!=L.end())
//		{
//			std::cout << "position: " << std::distance(L.begin(), result) << std::endl;
//			result++;
//		}
//		
//	}
//	
//	return 1;
//	
//	
//
//
//}

int main()
{
	cv::Mat A = (cv::Mat_<float>(3,1)<<1, 2, 3);
	cv::Mat B = (cv::Mat_<float>(3,1)<<3, 2, 1);
	cv::Mat C = (cv::Mat_<float>(3,3)<<1, 2, 3,4,5,6,7,8,9);

	cv::Mat CA = C*A;
	cv::Mat CB = C*B;

	double angle1 = acos(abs(A.dot(B) / (cv::norm(A)*cv::norm(B)))) * 180 / CV_PI;
	double angle2 = acos(abs(CA.dot(CB) / (cv::norm(CA)*cv::norm(CB)))) * 180 / CV_PI;
	double dist1 = cv::norm(A - B);
	double dist2 = cv::norm(CA - CB);

	return 1;
}
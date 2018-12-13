#pragma once
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include "g2o/core/robust_kernel_impl.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>

#include <chrono>

#include <iomanip>
#include <math.h>
#include <random>

using namespace Eigen;

class CurveFittingVertex :public g2o::BaseVertex<4, Vector4d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		CurveFittingVertex();
		virtual void setToOriginImpl()
	{
		_estimate << 0, 0, 0, 0;
	}

	virtual void oplusImpl(const double *_update)
	{
		Eigen::Map<const Vector4d> up(_update);
		_estimate += up;
	}

	bool read(std::istream& is) { return true; }
	bool write(std::ostream& os) const { return true; }
};

CurveFittingVertex::CurveFittingVertex() : g2o::BaseVertex<4, Eigen::Vector4d>()
{

}

//边

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		CurveFittingEdge();


	// 计算曲线模型误差
	void computeError()
	{
		const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
		const Vector4d abcd = v->estimate();
		double A = abcd[0], B = abcd[1], C = abcd[2], D = abcd[3];
		_error(0, 0) = _measurement - (A * sin(B*_x) + C * cos(D*_x));  // 误差函数：观测量减去估计量
	}

	//virtual void linearizeOplus();

	bool read(std::istream& is) { return true; }
	bool write(std::ostream& os) const { return true; }

public:
	double _x;
};

CurveFittingEdge::CurveFittingEdge() : g2o::BaseUnaryEdge<1, double, CurveFittingVertex>()
{

}
// Jacobin
//void CurveFittingEdge::linearizeOplus()
//{
//	CurveFittingVertex *vi = static_cast<CurveFittingVertex *>(_vertices[0]);
//	Vector4d abcd = vi->estimate();
//	double A = abcd[0], B = abcd[1], C = abcd[2], D = abcd[3];
//	// 误差项对待优化变量的Jacobin
//	_jacobianOplusXi(0, 0) = -sin(B*_x);
//	_jacobianOplusXi(0, 1) = -A*_x*cos(B*_x);
//	_jacobianOplusXi(0, 2) = -cos(D*_x);
//	_jacobianOplusXi(0, 3) = C*_x*sin(D*_x);
//}

class A 
{
public:
	A(int aa, int bb)
	{
		a = aa--;
		b = a*bb;
	}

	int a, b;
};
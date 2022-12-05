/*
	这一部分主要是为Odometry和Mapping过程中计算点到线的残差和点到面的残差来构建ceres所需
	在laserOdometry和laserMapping中利用特征来计算位姿变化时，需要计算残差，这里就是来计算残差的。应用了ceres 替代了手推高斯推导
	https://zhuanlan.zhihu.com/p/400014744
*/

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

/*
 * 对边缘特征匹配位姿进行估计的 Functor
 * 构建一个点对另外两个点组成的线的距离残差
 */
struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const  // 构建残差函数，点到直线距离，跟论文中一样
	{
		// 残差计算过程
		// 获取当前点 p 及上一帧的角点的 a, b
		// 将double数组转成eigen的数据结构，注意这里必须都写成模板
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		// 获取当前帧相对于上一帧的旋转，并根据给点的畸变因子进行去畸变
		// Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// 计算的是上一帧到当前帧的位姿变换，因此根据匀速模型，计算该点对应的位姿
		// 考虑运动补偿，ktti点云已经补偿过所以可以忽略下面的对四元数slerp插值以及对平移的线性插值
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		// 将当前点的位姿转换到上一帧结束的时间戳
		Eigen::Matrix<T, 3, 1> lp;
		// Odometry线程时，下面是将当前帧Lidar坐标系下的cp点变换到上一帧的Lidar坐标系下，然后在上一帧的Lidar坐标系计算点到线的残差距离
		// Mapping线程时，下面是将当前帧Lidar坐标系下的cp点变换到world坐标系下，然后在world坐标系下计算点到线的残差距离
		lp = q_last_curr * cp + t_last_curr;

		// 计算点 p 和 ab 的距离
		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);  // 两个向量叉乘的长度为这两个向量组成的平行四边形面积（两个三角形面积和）
		Eigen::Matrix<T, 3, 1> de = lpa - lpb; // 三角形的底边

		// 面积除以底边长度等于高的长度，即 p 点到直线 ab 距离
		// 这里的residual是三维的，但是论文中的残差是一维的，residual[0] = nu.norm() / de.norm();
		// 最终的残差本来应该是residual[0] = nu.norm() / de.norm(); 为啥也分成3个，我也不知
		// 道，从我试验的效果来看，确实是下面的残差函数形式，最后输出的pose精度会好一点点，这里需要
		// 注意的是，所有的residual都不用加fabs，因为Ceres内部会对其求 平方 作为最终的残差项
		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	// 使用ceres的自动求导
	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		// 添加新的观测值，构建残差块
		return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3> (new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
				                             //<,残差的维度,优化变量q的维度,优化变量t的维度>
	}

	// 观测量为当前帧的点 p，与上一帧构成边缘的点 a 和点 b以及畸变因子 s
	// 通过估计位姿 q 和 t，使得位姿变换后的 p 离线段 ab 距离最短
	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};


/*
 * 对平面特征进行位姿匹配估计的 Functor
 * 构建一个点对另外三个点组成的平面的距离残差
 */
struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		// 求出平面单位法向量
		// 点l、j、m就是搜索到的最近邻的3个点，下面就是计算出这三个点构成的平面ljlm的法向量
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();  // 对应的点到平面的距离只需要向这个归一化向量上投影就行了
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		// 对点进行去畸变
		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// 根据时间戳进行插值
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;  // 得到了相对于这一帧起始坐标的点云位置（去畸变的点云坐标）
		
		// 计算 p 到 ljm 平面的距离
		// d = pj 点乘 ljm 平面的单位法向量
		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	// 使用ceres的自动求导
	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	// 观测量为当前帧的点 p，上一帧中构成平面特征的三个点 j, l, m，以及 jlm 平面形成的法向量
	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;  // 畸变因子，0 表示起始点（对应上一帧结束时间点），1表示结束点（对应当前帧结束时间点）
};


/*
 * 对平面特征进行位姿匹配估计的 Functor
 * 利用法向量构建残差
 */
// 根据拟合的平面法向量，计算点到平面的距离作为误差函数
struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		// 将待匹配点转换至世界坐标系下
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

	    // 计算点到平面的距离，平面方程为 Ax + By + Cz + D = 0, negative_OA_dot_norm 表示常数项 D
		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		// 点(x0, y0, z0)到平面Ax + By + Cz + D = 0 的距离 = fabs(A*x0 + B*y0 + C*z0 + D) / sqrt(A^2 + B^2 + C^2)
		// 因为法向量（A, B, C）已经归一化了，所以距离公式可以简写为：距离 = fabs(A*x0 + B*y0 + C*z0 + D)
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};

/**
 * 构建两个点之间的距离残差
 */
// 没用到，被注释掉了
struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		// 将两个点转换至同一坐标系下然后计算其距离
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};
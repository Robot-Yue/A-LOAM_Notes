/*
	通过已经获得的激光里程计信息来消除激光里程计和地图之间的误差（也就是累计的误差），使得最终的姿态都是关于世界坐标的。
	进行帧与submap的点云特征配准，对Odometry线程计算的位姿进行微调
	https://zhuanlan.zhihu.com/p/400014744
*/

#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>  //互斥锁头文件
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"   //关于弧度和角度相互转换
#include "aloam_velodyne/tic_toc.h"  //计算时间间隔


int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

// 全局地图原点的栅格坐标，初始化在栅格中心，随着地图逐渐向某个方向边缘扩展
// 原点的坐标（以及地图）会逐渐往相反方向移动，以扩张地图在该方向的空间
// map子图的大小，cube的数量，即x,y,z方向各有21,21,11个cube
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
// 全局地图的尺寸，分辨率为 50m
// cube的总数量，也就是上图中的小格子个总数量 21 * 21 * 11 = 4851
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;

// 设置最大点云数量
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851

// 记录submap中的有效cube的index，注意submap中cube的最大数量为 5 * 5 * 5 = 125
int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

// input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
// 将三维栅格中的点云序列化存放在一维数组中
// 栅格坐标为 (i, j, k) 的点云存放位置为： i * laserCloudWidth + j * laserCloudHeight + k * laserCloudDepth
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

// kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

// 点云特征匹配时的优化变量
// 七维参数用于估计，优化参数为全局坐标系下的位姿（位置和姿态）
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
// Mapping线程估计的frame在world坐标系的位姿P,因为Mapping的算法耗时很有可能会超过100ms，所以
// 这个位姿P不是实时的，LOAM最终输出的实时位姿P_realtime,需要Mapping线程计算的相对低频位姿和
// Odometry线程计算的相对高频位姿做整合，详见后面laserOdometryHandler函数分析。此外需要注意
// 的是，不同于Odometry线程，这里的位姿P，即q_w_curr和t_w_curr，本身就是匹配时的优化变量。
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
// 下面的两个变量是world坐标系下的Odometry计算的位姿和Mapping计算的位姿之间的增量（也即变换，transformation）
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);


// 前端里程计估计的位姿到后端里程计估计的位姿的转换，（用于对里程计位姿进行修正）
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

// set initial guess
// 根据里程计的估计的位姿，和当前后端估计的位姿偏差，计算出全局坐标下的位姿
/*
 *  上一帧的增量wmap_wodom * 本帧Odometry位姿wodom_curr，旨在为本帧Mapping位姿w_curr设置一个初始值
 */
// 里程计位姿转化为地图位姿，作为后端初始估计
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

/*
 *    用在最后，当Mapping的位姿w_curr计算完毕后，更新增量wmap_wodom，
 *          旨在为下一次执行transformAssociateToMap函数时做准备
 */
// 更新odom到map之间的位姿变换
void transformUpdate()
{
	// 利用后端估计的世界位姿以及前端估计的世界位姿，更新位姿修正需要的旋转和平移
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

/*
 *     将Lidar坐标系下的点变换到map中world坐标系下
 *
 * @param      pi    原激光点
 * @param      po    变换后的激光点
 */
// 雷达坐标系点转化为地图点
void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

/*
 *      将map中world坐标系下的点变换到Lidar坐标系下，这个没有用到
 *
 * @param      pi    原激光点
 * @param      po    变换后的激光点
 */
// 地图点转化到雷达坐标系点
// 这个没有用到，是上面pointAssociateToMap的逆变换，即用Mapping的位姿w_curr，将world坐标系下的点变换到Lidar坐标系下
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

// 回调函数中将消息都是送入各自队列，进行线程加锁和解锁
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	// 互斥锁被锁定。线程申请该互斥锁，如果未能获得该互斥锁，则调用线程将阻塞(block)在该互斥锁上；
	// 如果成功获得该互诉锁，该线程一直拥有该互斥锁直到调用unlock解锁；如果该互斥锁已经被当前调用线程锁住，则产生死锁(deadlock)。
	mBuf.lock(); // 锁定 (线程加锁，避免线程冲突)
	cornerLastBuf.push(laserCloudCornerLast2); // 将所有 laserCloudCornerLast2 点加入到 cornerLastBuf 中
	mBuf.unlock(); // 解锁 (如果没有数据了，则线程解锁)
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

// receive odomtry
// 接受前端发送过来的里程计话题，并将位姿转换到世界坐标系下后发布
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock(); // 锁定 (线程加锁，避免线程冲突)
	odometryBuf.push(laserOdometry); // 将所有 laserOdometry 点加入到 odometryBuf 中
	mBuf.unlock(); // 解锁 (如果没有数据了，则线程解锁)

	// high frequence publish
		// 获取里程计位姿
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	// 里程计坐标系位姿转化为地图坐标系位姿
	// 对里程计估计的位姿根据目前的估计做出调整并发布，为高频消息，和里程计发布的频率一致
	// 为了保证 LOAM 整体的实时性，防止 Mapping 线程耗时 >100ms 导致丢帧，用上一次的增量 wmap_wodom 来更新
	// Odometry 的位姿，旨在用 Mapping 位姿的初始值（也可以理解为预测值）来实时输出，进而实现 LOAM 整体的实时性
	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 
	
	// 发布高频里程计数据
	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

// 主处理线程：进行Mapping，即帧与submap的匹配，对Odometry计算的位姿进行微调
void process()
{
	while(1)
	{
		// 四个队列分别存放边线点、平面点、全部点、和里程计位姿，要确保需要的buffer里都有值
		// laserOdometry模块对本节点的执行频率进行了控制，laserOdometry模块publish的位姿是10Hz，点云的publish频率没这么高，限制是2hz
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
			!fullResBuf.empty() && !odometryBuf.empty())
		{
			// 为了保证LOAM算法整体的实时性，Mapping线程每次只处理cornerLastBuf.front()及其他与之时间同步的消息
			mBuf.lock();  //线程加锁，避免线程冲突
 
			// 以cornerLastBuf为基准，把时间戳小于它的全部pop出去，保证其他容器的最新消息与cornerLastBuf.front()最新消息时间戳同步
			// odometryBuf只保留一个与cornerLastBuf.front()时间同步的最新消息
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			// 如果筛除完后 Buffer 为空，说明没有一组时间匹配的数据，因此退出这一次处理等待新消息
			if (odometryBuf.empty())
			{
				mBuf.unlock();  //如果没有数据了，则线程解锁
				break;
			}
			
			// surfLastBuf也只保留一个与cornerLastBuf.front()时间同步的最新消息
			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}
			
			// fullResBuf也如此也只保留一个与cornerLastBuf.front()时间同步的最新消息
			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
			
			// 确认收到的角点、平面、全点云以及里程计数据时间戳是否一致，由于前端每次会同时发布这四个数据，所以理论上应该可以得到四个时间戳相同的数据
			// 原则上取出来的时间戳都是一样的，如果不一样则说明有问题
			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			// 取出下一帧要处理的角点、平面以及全部 ROS 点云消息，转换为 PCL 格式
			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			// 取出上述消息对应的前端计算的当前位姿，存到对应的 q 和 t 中
			// 记录odometry前端线程传过来的位姿
			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			// 清空 cornerLastBuf 的历史缓存，为了LOAM 的整体实时性
			// 考虑到实时性，Mapping线程耗时>100ms导致的队列里缓存的其他边线点都pop出去，不然可能出现处理延时的情况
			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();
				printf("drop lidar frame in mapping for real time performance \n");
			}

			mBuf.unlock();

			TicToc t_whole;  //计算这个线程的全部时间
			// 根据前端结果，将里程计位姿转化为地图位姿得到后端优化的一个初始估计值
			transformAssociateToMap();

			TicToc t_shift; //计算位姿转换的时间
			
			// 后端地图本质上是一个以当前点为中心的一个珊格地图，根据初始估计值计算寻找当前位姿在地图中的索引，一个格子的边长是50m
			
			// 下面这是计算当前帧位置t_w_curr（在上图中用红色五角星表示的位置）IJK坐标（见上图中的坐标轴），
			// 参照LOAM_NOTED的注释，下面有关25呀，50啥的运算，等效于以50m为单位进行缩放，因为LOAM用1维数组
			// 进行cube的管理，而数组的index只用是正数，所以要保证IJK坐标都是正数，所以加了laserCloudCenWidth/Heigh/Depth
			// 的偏移，来使得当前位置尽量位于submap的中心处，也就是使得上图中的五角星位置尽量处于所有格子的中心附近，
			// 偏移laserCloudCenWidth/Heigh/Depth会动态调整，来保证当前位置尽量位于submap的中心处。

			// 计算当前位姿在全局三维栅格中存放的位置（索引），栅格分辨率为 50 
			// + 25.0 起四舍五入的作用，([0, 25) 取 0, [25, 50} 进 1)
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;  
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;
			
			// 如果小于25就向下取整，相当于四舍五入的一个过程
			// 由于计算机求余是向零取整，为了不使（-50.0,50.0）求余后都向零偏移，当被求余数为负数时求余结果统一向左偏移一个单位，也即减 1
			// 由于int(-2.1)=-2是向0取整，当被求余数为负数时求余结果统一向左偏移一个单位
			if (t_w_curr.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0)
				centerCubeK--;

			// 如果当前珊格索引小于3,就说明当前点快接近地图边界了，需要进行调整，相当于地图整体往x正方向移动

			// 当点云中心栅格坐标靠近栅格宽度负方向边缘时，需要将所有点云向宽度正方向移一个栅格，以腾出空间，保证栅格能够在宽度负方向容纳更多点云
			// 这里用 3 个栅格长度（150m）作为缓冲区域，即保证当前位姿周围 150 m 范围内的激光点都可以进行存放

			// 调整取值范围:3 < centerCubeI < 18， 3 < centerCubeJ < 18, 3 < centerCubeK < 8
			// 目的是为了防止后续向四周拓展cube（图中的黄色cube就是拓展的cube）时，index（即IJK坐标）成为负数
			// 如果处于下边界，表明地图向负方向延伸的可能性比较大，则循环移位，将数组中心点向上边界调整一个单位
			while (centerCubeI < 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{ 
						// 宽度方向上的边界（最后一个点云） 
						int i = laserCloudWidth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						
						// 不改变点云在高度和深度的栅格坐标，对其在宽度方向向往正方向移动一个单位，正方向边缘的点云会被移除
						for (; i >= 1; i--)// 在I方向上，将cube[I] = cube[I-1],最后一个空出来的cube清空点云，实现IJK坐标系向I轴负方向移动一个cube的
										   // 效果，从相对运动的角度看，就是图中的五角星在IJK坐标系下向I轴正方向移动了一个cube，如下面的动图所示，所
				                           // 以centerCubeI最后++，laserCloudCenWidth也会++，为下一帧Mapping时计算五角星的IJK坐标做准备。
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI++;
				// 在移动完点云后，需要将原点对应的栅格坐标也向宽度正方向移动一个单位，保证栅格坐标计算的正确性
				laserCloudCenWidth++;
			}

			// 当点云中心栅格坐标靠近栅格宽度正方向边缘时，需要将所有点云向宽度负方向移一个栅格，进行和上面一样的操作
			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// 确定宽度方向上的边界
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}

			// 对栅格高度方向进行一样的操作
			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}
			// 对栅格高度方向进行一样的操作
			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			// 对栅格深度方向进行一样的操作
			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}
			// 对栅格深度方向进行一样的操作
			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}

			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;

			// 向IJ坐标轴的正负方向各拓展2个cube，K坐标轴的正负方向各拓展1个cube，上图中五角星所在的蓝色cube就是当前位置
			// 所处的cube，拓展的cube就是黄色的cube，这些cube就是submap的范围
			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
						{ 
							// 记录submap中的有效cube的index,（在所有cube中submap的cube的index）
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			// 提取局部地图点云（当前载体附近的点云）
			laserCloudCornerFromMap->clear();
			laserCloudSurfFromMap->clear();
			
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				// 将有效index的cube中的点云叠加到一起组成submap的特征点云
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

			
			// 对待处理的角点点云进行降采样处理
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
			// 对待处理的平面点云进行降采样处理
			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
			
			// 到此整个子图构建结束，接下来计算特征和子图的匹配
			printf("map prepare time %f ms\n", t_shift.toc());
			printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
			
			// 当局部地图中有足够的角点和平面点才进行匹配和优化操作
			// 第一帧点云还没加入到cube中，此时不会满足这一条件
			if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)
			{
				TicToc t_opt;
				TicToc t_tree;

				// 用局部地图中的角点和平面生成 KD 树
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
				kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				printf("build tree time %f ms \n", t_tree.toc());

				for (int iterCount = 0; iterCount < 2; iterCount++)
				{
					//ceres::LossFunction *loss_function = NULL;
					// 初始化图，损失函数，设置定鲁棒核函数
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;
					
					// 加入要优化的参数块，这里分别是姿态（4维）和位置（3维）
					ceres::Problem problem(problem_options);
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;
					int corner_num = 0;

					// 对Edgepoints构建误差方程
					// 对将要匹配中的每个角点进行以下操作
					for (int i = 0; i < laserCloudCornerStackNum; i++)
					{
						pointOri = laserCloudCornerStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						// 需要注意的是submap中的点云都是world坐标系，而当前帧的点云都是Lidar坐标系，所以
						// 在搜寻最近邻点时，先用预测的Mapping位姿w_curr，将Lidar坐标系下的特征点变换到world坐标系下
						pointAssociateToMap(&pointOri, &pointSel);	// 将当前点从激光雷达坐标系转到世界（map）坐标系下
						// 在submap的corner特征点（target）中，寻找距离当前帧corner特征点（source）最近的5个点
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

						// 当搜索到的 5 个点都在当前选择点的 1m 范围内，才进行以下操作
						if (pointSearchSqDis[4] < 1.0)
						{ 
							// 获取这 5 个点的位置并计算其平均值作为线段中心
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								nearCorners.push_back(tmp);
							}
							// 计算这个5个最近邻点的中心
							center = center / 5.0;

							// 根据距离设置这 5 个点的协方差矩阵 cov = sum(mean * mean^T)
							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							for (int j = 0; j < 5; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
							}

							// 协方差矩阵是厄米特矩阵，计算其特征值和特征向量
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

							// if is indeed line feature
							// note Eigen library sort eigenvalues in increasing order
							// 计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布，此为PCA的原理
							// 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向矢量
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							
							// 如果最大的特征值 >> 其他特征值，则5个点确实呈线状分布，否则认为直线“不够直”
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
							{ 
								// 在这 5 个点的中心，根据直线（特征向量）方向计算出两个点的坐标作为匹配点
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;
								// 从中心点沿着方向向量向两端移动0.1m，使用两个点代替一条直线，这样计算点到直线的距离的形式就跟laserOdometry中保持一致
								point_a = 0.1 * unit_direction + point_on_line;
								point_b = -0.1 * unit_direction + point_on_line;
								
								// 这里点到线的ICP过程就比Odometry中的要鲁棒和准确一些了（当然也就更耗时一些）
					            // 因为不是简单粗暴地搜最近的两个corner点作为target的线，而是PCA计算出最近邻的5个点的主方向作为线的方向，而且还会check直线是不是“足够直”
								
								// 将两个点以及当前选择的角点（注意是在点云中的，坐标是在载体坐标系下表示）传入 ceres 中构建残差
								ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								corner_num++;	
							}							
						}

						// 如果这个 5 个点离选择的点很近，则考虑为同一个点，
						// 计算中心点并将其和当前选择的点传入 ceres 中，最小化这两个点的距离
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					// 对Planar Points构建误差方程
					// 对将要匹配中的每个平面点进行以下操作
					int surf_num = 0;
					for (int i = 0; i < laserCloudSurfStackNum; i++)
					{
						// 获取原始点并将其坐标转换至世界坐标系，在平面点的 KD 树搜索最近的 5 个点
						pointOri = laserCloudSurfStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

						// 求面的法向量就不是用的PCA了（虽然论文中说还是PCA），使用的是最小二乘拟合
						// 假设平面不通过原点，则平面的一般方程为Ax + By + Cz + 1 = 0，用这个假设可以少算一个参数，提效
						Eigen::Matrix<double, 5, 3> matA0;
						Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
						// 用上面的2个矩阵表示平面方程就是 matA0 * norm（A, B, C） = matB0，这是个超定方程组，因为数据个数超过未知数的个数

						// 当搜索到的 5 个点都在当前选择点的 1m 范围内，才进行以下操作
						if (pointSearchSqDis[4] < 1.0)
						{
							// 将 5 个点按列插入 5x3 矩阵中
							// 设平面方程为ax+by+cz+d=0，取d=1，则（a,b,c）为平面法向量
							for (int j = 0; j < 5; j++)
							{
								matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								//printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
							}
							// find the norm of plane
							// 计算平面法向量，平面方程为 A'x + B'y + C'z = -1，代入 5 个点，利用最小二乘法解出参数即可得到法向量
							Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);  
							// Ax + By + Cz + 1 = 0，全部除以法向量的模长，方程依旧成立，而且使得法向量归一化了
							double negative_OA_dot_norm = 1 / norm.norm();  // 法向量长度的倒数
							norm.normalize();  // 法向量归一化，相当于直线Ax + By + Cz + 1 = 0两边都乘negative_OA_dot_norm

							// Here n(pa, pb, pc) is unit norm of plane
							// 通过计算每个点到平面的距离来判断平面拟合的效果，距离公式为：d = fabs(Ax0 + By0 + Cz0 + D) / sqrt(A^2 + B^2 + C^2)
							// 归一化后平面公式为：Ax + By + Cz + D = 0, D = 1/sqrt(A'^2+B'^2+C'^2)
							// 因此，计算公式为：d = Ax + By + Cz + negative_OA_dot_norm
							bool planeValid = true;
							for (int j = 0; j < 5; j++)
							{
								// if OX * n > 0.2, then plane is not fit well
								// 判断法向量的拟合效果，即拟合原方程ax+by+cz+1=0的效果
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									// 平面没有拟合好，平面“不够平”
									planeValid = false;
									break;
								}
							}
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							if (planeValid)
							{
								// 将选择的点和法向量传入 ceres 中构建残差块
								// 构造点到面的距离残差项，同样的，具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
								ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								surf_num++;
							}
						}

						// 如果这 5 个点都离选择点很近，考虑为同一个点，将其中心和待选择的点传入 ceres 中最下化其距离
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
													laserCloudSurfFromMap->points[pointSearchInd[j]].y,
													laserCloudSurfFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					printf("mapping data assosiation time %f ms \n", t_data.toc());

					// 对整个图（包括平面点和角点）进行优化
					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					printf("mapping solver time %f ms \n", t_solver.toc());

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				printf("mapping optimization time %f \n", t_opt.toc());
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");
			}
			// 完成ICP（迭代2次）的特征匹配后，用最后匹配计算出的优化变量w_curr，更新增量wmap_wodom，为下一次Mapping做准备
			transformUpdate();


			TicToc t_add;
			// 将当前帧的（次极大边线点云，经过降采样后的）存入对应的边线点云的cube
			// 下面两个for循环的作用就是将当前帧的特征点云，逐点进行操作：转换到world坐标系并添加到对应位置的cube中
			for (int i = 0; i < laserCloudCornerStackNum; i++)
			{
				// 对当前新加入的角点点云中的所有点，进行坐标转换至世界坐标系下，找到其对应的点云栅格坐标
				// 插入到该栅格中存放的点云中
				// Lidar坐标系转到world坐标系
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

				// 计算本次的特征点的IJK坐标，进而确定添加到哪个cube中
				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}

			// 将当前帧的（次极小平面点云，经过降采样后的）存入对应的平面点云的cube
			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				// 对当前新加入的平面点点云中的所有点，进行坐标转换至世界坐标系下，找到其对应的点云栅格坐标
				// 插入到该栅格中存放的点云中
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			printf("add points time %f ms\n", t_add.toc());

			// 因为cube中新增加了点云，之前已经存有点云的cube的密度会发生改变，这里全部重新进行一次降采样
            // 这个地方可以简单优化一下：如果之前的cube没有新添加点云就不需要再降采样：
			// 这可以通过记录上一个循环中存入对应cubeInd的index来实现
			TicToc t_filter;
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				// 记录submap中的有效cube的index（在所有cube中submap的cube的idx）
				int ind = laserCloudValidInd[i];

				// 加完新特征点后对地图里的特征点进行下采样
				// 对这次处理涉及到的点云进行降采样处理
				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			//publish surround map for every 5 frame
			// 每 5 帧发布（更新）一次局部地图
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				for (int i = 0; i < laserCloudSurroundNum; i++)  //laserCloudSurroundNum应该是5*5*3的
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			// 每 20 帧发布（更新）一次全局地图（将所有点云相加并发布）
			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}
			
			// 将当前帧点云（所有点，不限于角点和平面点）转换至世界坐标系下发布
			// 将激光点转换到世界坐标系下并发布，可视化用
			int laserCloudFullResNum = laserCloudFullRes->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			printf("mapping pub time %f ms \n", t_pub.toc());

			printf("whole mapping time %f ms +++++\n", t_whole.toc());

			// 发布更新后的全局位姿和路径（在后端建立的全局坐标系下）
			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);  // 发布地图坐标系下的位姿

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);  // 发布地图坐标系下的位姿组成的路径

			// 设置初始位姿（起始点）到当前位姿的转换关系，并发布
			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

			frameCount++;
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{
	// 初始化节点
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	// 获取参数：线和平面的分辨率
	float lineRes = 0;  // 次极大边线点集体素滤波分辨率
	float planeRes = 0;  // 次极小平面点集体素滤波分辨率
	// 对地图特征点云下采样的分辨率
	nh.param<float>("mapping_line_resolution", lineRes, 0.4); 
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);

	// 根据给定的线和平面的分辨率对角点和平面的降采样滤波最小尺寸进行设置
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

	// 初始化 subcriber，订阅角点和平面的点云消息、前端里程计的估计结果以及全部点云
	// 订阅从odometry前端发来的4个topic消息，分别放在各自的buffer里
	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler); // 从laserOdometry节点接收次极大边线点
	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler); // 从laserOdometry节点接收次极小平面点
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler); // 从laserOdometry节点接收到的最新帧的位姿T_cur^w
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler); // 从laserOdometry节点接收到的原始点云（只经过一次降采样）

	// 初始化 publishers，发布载体附近的点云（作为局部观测）、当前建立的点云地图、当前注册的点云、后端生成的里程计以及路径
	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100); // submap所在cube中的点云
	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100); // 所欲cube中的点云
	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100); // 原始点云
	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100); // Mapping模块优化之后的T_{cur}^{w}，未经过位姿融合
	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100); // Transform Integration之后的高频位姿数据
	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100); // Mapping模块优化之后的T_{cur}^{w}构成的轨迹

	// 重置全局点云（包括角点以及平面）
	for (int i = 0; i < laserCloudNum; i++)
	{
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	// 初始化建图线程
	std::thread mapping_process{process};

	ros::spin();

	return 0;
}
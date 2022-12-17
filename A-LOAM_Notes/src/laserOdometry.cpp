/*
    通过读取scanRegistration.cpp中的信息来计算帧与帧之间的变化，最终得到里程计坐标系下的激光雷达的位姿。
    laserOdometry 订阅了5个话题：有序点云、极大平面点、次极小平面点、极小平面点、次极小平面点。
                  发布了4个话题：有序点云、上一帧的平面点、上一帧的边线点、当前帧位姿粗估计
    https://zhuanlan.zhihu.com/p/396331443
*/

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0  // 表示激光点云是否已经被去过畸变

int corner_correspondence = 0, plane_correspondence = 0;

// constexpr表达式是指值不会改变并且在编译过程就能得到计算结果的表达式
constexpr double SCAN_PERIOD = 0.1;  // 激光雷达的频率，0.1s
constexpr double DISTANCE_SQ_THRESHOLD = 25;  // KDTree搜索时相关的阈值，找最近点的距离平方的阈值
constexpr double NEARBY_SCAN = 2.5;  // 对应论文中Fig.7找临近点时的nearby_scan的范围找点时最远激光层的阈值

int skipFrameNum = 5;                             // 控制建图的频率，通过.launch文件设定
bool systemInited = false;                        // 系统是否初始化，主要用来跳过第一帧

double timeCornerPointsSharp = 0;                 // 读取queue中数据时，存储时间戳的，没必要弄成全局变量
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());  //设定kd-tree的智能指针
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
 
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// 从当前框架到世界框架的转变
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);  // 激光雷达在世界坐标系中的位姿，用四元数来表示方向
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
// 点云特征匹配时的优化变量
double para_q[4] = {0, 0, 0, 1};  // ceres用来优化时的数组，四元数
double para_t[3] = {0, 0, 0};  // 平移

// 下面的2个分别是优化变量para_q和para_t的映射：表示的是两个world坐标系下的位姿P之间的增量，例如△P = P0.inverse() * P1
// 优化变量para_q、para_t的映射(共享内存)，这两个变量的含义很重要
// 这两个变量就是论文中的优化变量T_{k+1}^L的逆
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// 将消息缓存到对应的queue中，以便后续处理
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

// 激光点去畸变，把所有的点补偿到起始时刻
// undistort lidar point
// TransformToStart() 当前点云中的点相对第一个点去除因匀速运动产生的畸变，效果相当于得到在点云扫描开始位置静止扫描得到的点云
void TransformToStart(PointType const *const pi, PointType *const po)  // 根据时间进行插值，得到去畸变的点云位置
{
    // 插值比
    double s;
    //由于kitti数据集上的lidar已经做过了运动补偿，因此这里就不做具体补偿了
    if (DISTORTION)
        // intensity 实数部分存的是 scan上点的 id ;虚数部分存的这一点相对这一帧起始点的时间差
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD; // s代表要转换的点的时间戳占比                                                              
    else
        s = 1.0;  //s = 1s说明全部补偿到点云结束的时刻
    //s = 1;
    
    // 这里相当于是一个匀速模型的假设，把位姿变换分解成平移和旋转
    // 四元数插值有线性插值和球面插值，球面插值更准确，但是两个四元数差别不大时，二者精度相当 
    // q_last_curr：读作curr到last的 q，也就是curr在last坐标系下的旋转和位置，t同理
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr); // slerp函数(球面线性插值)
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z); //把当前点的坐标取出
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;  // 通过旋转和平移将当前点转到帧起始时刻坐标系下的坐标
   
    // 将求得的转换后的坐标赋值给输出点
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity; // 赋值原的intensity
}

// transform all lidar points to the start of the next frame
// 反变换的思想，首先把点统一到起始时刻坐标系下，再通过反变换，得到结束时刻坐标系下的点
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    // 首先做畸变校正,都转到起始时刻
    pcl::PointXYZI un_point_tmp; // 转到帧起始时刻坐标系下的点
    TransformToStart(pi, &un_point_tmp); // 先统一到起始时刻

    //再通过反变换的方式，将起始时刻坐标系下的点转到结束时刻坐标系下 
    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z); // 取出起始时刻坐标系下的点
    //q_last_curr \ t_last_curr 是结束时刻坐标系转到起始时刻坐标系的 旋转和平移        参考：https://blog.csdn.net/qq_32761549/article/details/120738568
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);  //通过反变换，求得转到结束时刻坐标系下的点坐标

    //将求得的转换后的坐标赋值给输出点
    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();
    //Remove distortion time info
    //移除掉 intensity 相对时间的信息
    po->intensity = int(pi->intensity);
}

// 操作都是送去各自的队列中，加了线程锁以避免线程数据冲突
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    // ***话题的订阅与发布***
    // 初始化里程计节点
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
    // if 1, do mapping 10 Hz, if 2, do mapping 5 Hz. Suggest to use 1, it will adjust frequence automaticlly
    // 设定里程计的帧率（每跳过skipFrameNum帧采样1帧）
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);  // 前端计算的频率，launch默认设置的为1，表示10Hz
    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    // 从scanRegistration订阅的消息话题
    // 极大边线点  次极大边线点  极小平面点  次极小平面点 全部有序点云点
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    // 发布上一帧的边线点
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    // 发布上一帧的平面点
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    // 发布全部有序点云，就是从scanRegistration订阅来的点云，未经其他处理
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    // 发布帧间的位姿变换
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 发布帧间的平移运动
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);  // 100ms 执行一次，频率为 10 Hz



    // ***帧间匹配与位姿估计***
    while (ros::ok()) // 重复执行该循环
    {
        ros::spinOnce(); // 等待回调函数执行完毕，执行一次后续代码

        // 如果所有订阅的话题都收到了，有一个队列为空都不行(如果各个特征点云的缓冲区不为空则进行处理)
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() && !surfFlatBuf.empty() && !surfLessFlatBuf.empty() && !fullPointsBuf.empty())
        {
            // 这里是5个queue记录了最新的极大/次极大边线点，极小/次极小平面点，全部有序点云
            // 用队列头的时间戳，分配时间戳
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            // 如果时间不同步，则报错
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                // ROS出现丢包会导致这种情况
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock(); // 对 buffer 加锁防止其他线程（subscriber）对 buffer 进行操作
            // 从队列中取出来对应的数据，并转换成 pcl 点云数据
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();  // 对 buffer 解锁

            TicToc t_whole;
            // initializing 和 特征点匹配位姿估计
            // 对初始帧的操作是，跳过位姿估计，直接作为第二帧的上一帧
            if (!systemInited)  // 主要用来跳过第一帧数据
                                // 仅仅将cornerPointsLessSharp保存至laserCloudCornerLast
                                // 将surfPointsLessFlat保存至laserCloudSurfLast，以及更新对应的kdtree
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            
            // ***主要的处理部分***
            else
            {
                // 极大边线点的数量
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                // 极小平面点的数量
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt;
                // 点到线以及点到面的非线性优化，迭代2次（选择当前优化的位姿的特征点匹配，并优化位姿（4次迭代），然后重新选择特征点匹配并优化）
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;  //初始化 角点匹配的点对的数量
                    plane_correspondence = 0;  //初始化 面点匹配的点对的数量

                    // http://www.ceres-solver.org/nnls_modeling.html?highlight=loss#_CPPv2N5ceres12LossFunctionE
                    //ceres::LossFunction *loss_function = NULL;
                    // 使用ceres核函数Huber来减少外点的影响（筛除外点、无效点的方法）
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1); // 当残差大于0.1时将其权重减小; 定义一个0.1阈值的huber核函数，优化时抑制离群点用
                    // 由于旋转不满足一般意义的加法，因此这里使用ceres自带的local param
                    // http://www.ceres-solver.org/nnls_modeling.html?highlight=eigenquaternionparameterization#_CPPv2N5ceres31EigenQuaternionParameterizationE
                    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();  // 进行四元数的参数化
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    // 待优化的变量是帧间位姿，平移和旋转，这里旋转使用四元数来表示
                    problem.AddParameterBlock(para_q, 4, q_parameterization);  // 设定优化参数的自变量，q_curr_last(x, y, z, w)和t_curr_last，四元数旋转和平移
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;



                    TicToc t_data;
                    // find correspondence for corner features
                    // ***为角点特征找到对应关系***
                    // 基于最近邻原理建立corner特征点之间关联，每一个极大边线点去上一帧的次极大边线点集中找匹配
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        // 这个函数类似论文中公式（5）的功能
                        // 将当前帧的极大边线点（记为点O_cur），变换到上一帧Lidar坐标系（记为点O），以利于寻找极大边线点的correspondence
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel); // 将激光点转换到这一帧起始时刻的坐标系下（根据时间线性插值粗略的去畸变）
                                                                                      // 注意这里去畸变只是为了方便找最近点，后面传给cost_function的为原始点，在cost_function里根据优化来进行去畸变
                        // 在点云中寻找点searchPoint：pointSel的k近邻的值，返回下标pointSearchInd和距离pointSearchSqDis，k为1就是最近邻搜索
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); // kdtree中的点云是上一帧的corner_less_sharp，所以这是在上一帧
                                                                                                        // 的corner_less_sharp中寻找当前帧corner_sharp特征点O的最近邻点（记为l）

                        int closestPointInd = -1, minPointInd2 = -1;
                        // 如果最近邻的corner特征点之间距离平方小于阈值，则最近邻点l有效
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0]; // 对应的最近距离的index取出来
                            // 找到了目标点 i 的最邻近点 l 的id
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity); // 最近点在scan第几层（intensity的整数部分）
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;


                            // search in the direction of increasing scan line
                            // 寻找点O的另外一个最近邻的点j
                            // 在点 l 的两侧寻找附近扫描线上的最邻近点 j
                            // laserCloudCornerLast是上一帧的corner_less_sharp特征点，由于提取特征时是按照scan的顺序提取的
                            // 所以laserCloudCornerLast中的点也是按照scanID递增的顺序存放的
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                // 要求不在同一层激光线束上第二个最近点
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                // 要求找到的线束距离当前线束不能太远
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                // 计算该点和当前找到的角点之间的距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) * (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) * (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2) 
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            // 同样的再反方向(下)寻找对应角点
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) * (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) * (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;  // 第二个最近邻点有效，更新点j
                                    minPointInd2 = j;
                                }
                            }
                        }
                        // 即特征点O的两个最近邻点l和j都有效
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            // 取出特征点O和两个最近邻点 l 和 j
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;  // 运动补偿系数，kitti数据集的点云已经被补偿过，所以s = 1.0
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            
                            // 用点O，l，j构造点到线的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下：残差 = 点O到直线lj的距离
                            // 具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }


                    // find correspondence for plane features
                    // ***为平面特征点找到对应关系***
                    // 与上面的建立corner特征点之间的关联类似，寻找平面特征点O的最近邻点l、j、m，即基于最近邻原理建立surf特征点之间的关联
                    for (int i = 0; i < surfPointsFlatNum; ++i)  // 所有的Planar Points
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);  // 将当前平面点位姿转换至当前帧起始时间
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);  // 先在上一帧的平面点中找到距离当前点最近的点

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;  // 然后找到3个最近点，计算到这3个点拟合的平面的距离，作为误差函数
                        // 距离必须小于给定阈值
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)  // 找到的最近邻点l有效
                        {
                            // 取出找到的上一帧面点的索引
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            // 取出最近的面点在上一帧的第几根scan上面
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            // 额外再寻找两个点，要求：一个点和最近点同一个scan，另一个是不同的scan
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                // 不能和当前找到的上一帧面点约束距离太远
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                // 计算和当前帧该点距离
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) * (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) * (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                // 其中一个次最近点需要在同一层或更底层
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;  // 找到的第2个最近邻点有效，更新点j，注意如果scanID准确的话，一般点l和点j的scanID相同
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                // 另一个次最近点需要在更高层，这样选出来的3个点不容易共线，可以更好的拟合出一个平面
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;  // 找到的第3个最近邻点有效，更新点m，注意如果scanID准确的话，一般点l和点j的scanID相同，且与点m的scanID不同
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            // 同样的方式，去按照降序方向寻找这两个点
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // 即特征点O的三个最近邻点l、j、m都有效
                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                           surfPointsFlat->points[i].y,
                                                           surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                             laserCloudSurfLast->points[closestPointInd].y,
                                                             laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                             laserCloudSurfLast->points[minPointInd2].y,
                                                             laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                             laserCloudSurfLast->points[minPointInd3].y,
                                                             laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)  // 如果激光点云没有做过去畸变，这里用s来计算需要插值的比例（时间）
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                // 用点 O，l，j，m 构造点到面的距离的残差项，注意这三个点都是在上一帧的Lidar坐标系下，即，残差 = 点O到平面ljm的距离
                                // 同样的，具体到介绍lidarFactor.cpp时再说明该残差的具体计算方法
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());
                    // 如果总的约束太少，就打印一下
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }
                    
                    
                    // 调用ceres求解器求解，对位姿进行优化使得残差最小
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());
                }
                printf("optimization twice time %f \n", t_opt.toc());  // 经过两次LM优化消耗的时间
                
                // 用最新计算出的位姿增量，更新帧间匹配的结果，得到激光雷达的世界坐标系
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;  // 相当于是t_w_curr = t_w_last + q_w_last * t_last_curr
                q_w_curr = q_w_curr * q_last_curr;  // 相当于是q_w_curr = q_w_last * q_last_curr
            }

            TicToc t_pub;

            // publish odometry
            // 发布lidar里程计结果
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            // 以四元数和平移向量发出去
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);  // 发布激光里程计的位姿

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);  // 发布激光里程计的路径

            // 对应论文Algorithm 1: Lidar Odometry中的最后一步，将P_{k+1}帧的点云投影到t_{k+2}时刻
            // transform corner features and plane features to the scan end point
            if (0)  // 如果点云存在畸变，则将特征点转换到下一帧的开始时间。相当于是对上一帧特征点进行了去畸变处理，方便跟下一帧进行匹配
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            // 用当前的角点和平面点点云更新上一帧信息
            // 位姿估计完毕之后，当前边线点和平面点就变成了上一帧的边线点和平面点，
            // 把索引和数量都转移过去
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
            // kdtree设置当前帧，用来下一帧lidar odom使用
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
            
            // 控制输出的频率，由此决定mapping线程的频率(为了控制最后一个节点的执行频率)
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;
                
                // 发布次极大边线点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                // 发布次极小平面点
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                // 原封不动的转发上一个node处理出（简单滤波）的当前帧点云）
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}

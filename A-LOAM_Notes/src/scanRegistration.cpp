/*
    用来读取激光雷达数据并且对激光雷达数据进行整理，去除无效点云以及激光雷达周围的点云
    并且计算每一个点的曲率，按照提前设定好的曲率范围来分离出四种不同类型的点
    https://zhuanlan.zhihu.com/p/400014744
*/

#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
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

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;  // 激光雷达转一圈的频率10Hz，默认为0.1s
const int systemDelay = 0;  // 系统接收雷达的延迟，接收到的雷达帧数量systemInitCount > systemDelay，才初始化成功
int systemInitCount = 0;    // 接收到的雷达帧的数量通过这个变量表示
bool systemInited = false;  // systemInitCount > systemDelay后，systemInited = true
int N_SCANS = 0;  // 雷达线数16、32、64，从launch文件中读入
float cloudCurvature[400000];  // 每个点云的曲率大小,全局数组的大小要足够容纳一帧点云
int cloudSortInd[400000];      // 每个点云的index，根据曲率进行排序的时候使用
int cloudNeighborPicked[400000];  // 表示某点已经被打上过标签的标记 (避免特征点密集分布，每选一个特征点其前后5个点**一般**就不选取了)
int cloudLabel[400000];   // 记录特征点属于那种类型：极大边线点、次极大边线点、极小平面点、次极小平面点
                                              /* Label 2: corner_sharp
                                                 Label 1: corner_less_sharp, 包含Label 2                     
                                                 Label -1: surf_flat
                                                 Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样
                                              */
/**
 *                sort的升序比较函数
 * 1、std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
 * 2、使用快排对某个范围内的特征点按照点的曲率进行升序排序
 * 3、cloudSortInd[]数组就是用来存储排好序的特征点的ID的
 */
// 根据曲率进行排序的比较函数
bool comp (int i,int j) { 
    return (cloudCurvature[i]<cloudCurvature[j]); 
}

ros::Publisher pubLaserCloud;              // 发布原始点云（经过无序-->有序的处理）
ros::Publisher pubCornerPointsSharp;       // 发布极大边线点
ros::Publisher pubCornerPointsLessSharp;   // 发布次极大边线点
ros::Publisher pubSurfPointsFlat;          // 发布极小平面点
ros::Publisher pubSurfPointsLessFlat;      // 发布次极小平面点
ros::Publisher pubRemovePoints;            // 没用上
std::vector<ros::Publisher> pubEachScan;   // 发布雷达的每条线扫

bool PUB_EACH_LINE = false;  // 是否发布每条线扫
double MINIMUM_RANGE = 0.1;  // 雷达点可距雷达原点的最近距离阈值，小于该距离的点将会被去掉，可以在launch文件中进行设置



/**
 *            去除过近距离的点云
 * @param[in]  cloud_in   The cloud in
 * @param      cloud_out  The cloud out
 * @param[in]  thres      The thres，激光点距离雷达中心的距离小于这个阈值就被去掉
 * @tparam     PointT     { PCL点的类型模板 }
 */
// ****雷达周边过近点移除（通常这些点被认为是不可靠的）****
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 假如输入输出点云不存在同一个变量，则需要将输出点云的时间戳和容器大小与输入点云同步
    // 初始化返回点云，预先分配空间以避免添加的点的时候造成频繁的空间分配
    // If the clouds are not the same, prepare the output
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0; // 表示C中任何对象所能达到的最大长度，它是无符号整数；在64位系统上定义为unsigned long，也就是64位无符号整型；
                  // size_t的目的是提供一种可移植的方法来声明与系统中可寻址的内存区域一致的长度

    // 把距离雷达原点小于给定阈值的点云去除,也就是距离雷达过近的点去除
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        // Resize to the correct size
        cloud_out.points.resize(j);
    }

    // 这几个变量手册上说是Mandatory，必须要设定的；KITTI是无序的点云
    // 过滤掉之后每一条线的激光点云数量不一定，所以没有办法通过宽和高区分线，因此这里不做特殊处理
    // 这里是对每条扫描线上的点云进行直通滤波，因此设置点云的高度为1，宽度为数量，稠密点云
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j); //static_cast可使用于需要明确隐式转换的地方，c++中用static_cast用来表示明确的转换
    cloud_out.is_dense = true;  // 将去除NaN后的点云设置为dense点云
}



/**
 *          "/velodyne_cloud"话题的回调函数
 *1、无序点云-->有序点云
 *2、提取特征点
 * @param[in]  laserCloudMsg  The laser cloud message
 */
// ****激光处理回调函数（点云来一次调一次）****
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    // 首先判断是否初始化，systemDelay可以自己设置，用来丢掉前几帧
    // 如果系统没有初始化，就等几帧
    // 作用就是延时，为了确保有IMU数据后, 再进行点云数据的处理
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }
    
    //作者自己设计的计时类，以构造函数为起始时间，以toc()函数为终止时间，并返回时间间隔(ms)
    TicToc t_whole; // 总时间
    TicToc t_prepare; // 预处理时间
    
    //每条扫描线上的可以计算曲率的点云，点的起始索引和结束索引
    //分别用scanStartInd数组和scanEndInd数组存储每条线对应的起始和结束索引
    std::vector<int> scanStartInd(N_SCANS, 0);  // 定义了16个整型变量，每个变量的初值为1
    std::vector<int> scanEndInd(N_SCANS, 0);

    // 把点云消息从ros格式转到pcl格式
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;  // 输入点云，pcl点云格式
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);  // 将传入的ros消息格式转为pcl库里的点云格式
    
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);  // ***利用pcl的函数首先对点云滤波，去除点云中的NaN（无效）点*** 函数有三个参数:输入点云，输出点云，对应保留的索引
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);  // 自己设计的函数，去除太近的点，其中removeClosedPointCloud函数的代码风格是仿照pcl中removeNaNFromPointCloud的
                                                                        // 用来去除离激光雷达很近的点（这样的点很可能是机器人自身的，没有匹配意义），即去除距离小于阈值的点
    int cloudSize = laserCloudIn.points.size();
    
    

    // ***计算点云起始点和结束点的朝向（角度）***
    
    // atan2()默认返回逆时针角度，由于LiDAR通常是顺时针扫描，所以往往使用-atan2( )函数
    // 理论上起始点和结束点的差值应该是 0，为了显示两者区别，将结束点的方向补偿 2pi
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);  // atan2()函数的语法：atan2(y, x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;  //atan2范围是[-PI,PI],这里加上2PI是为了保证起始到结束相差2PI符合实际

    // 处理几种个别情况，以保持结束点的朝向和起始点的方向差始终在 [pi, 3pi] 之间 （实际上是 2pi 附近）
    if (endOri - startOri > 3 * M_PI)
    {
        // case 1: 起始点在 -179°，结束点在 179°，补偿 2pi 后相差超过一圈，实际上是结束点相对于起始点还没转完整一圈
        // 因此将结束点的 2pi 补偿去掉后为 179°，与起始点相差 358°，表示结束点和起始点差别是一圈少2°
        endOri -= 2 * M_PI; //额外减去2PI
    }
    else if (endOri - startOri < M_PI)
    {
        // case 2: 起始点为 179°，结束点为 -179°，补偿后结束点为 181°，此时不能反映真实差别，需要
        // 对结束点再补偿上 2pi，表示经过了一圈多 2°
        endOri += 2 * M_PI; //额外补偿2PI
    }
    //printf("end Ori %f\n", endOri);



    // ***计算每一个点对应的线，计算每一个点相对于第一个点的时间并重新存储***
    bool halfPassed = false;
    int count = cloudSize;  // 去除掉一些非法点之后的点云数量
    PointType point; //typedef pcl::PointXYZI PointType; //typedef定义了一种类型的新别名
    // 将点云按扫描线分别存储在一个子点云（laserCloudScans的数组）中，并计算每个点在该线束中根据起始和终点计算的位置（时间）
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);

    for (int i = 0; i < cloudSize; i++)
    {
        // 小技巧：对临时变量 Point 只初始化一次减少空间分配
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        // ***计算垂直视场角：告诉你是第几根scan线，决定这个激光点所在的scanID***
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;  // M_PI 在#include<math.h>中定义为：π = 3.1415926...
        int scanID = 0;

        // 基于不同线的激光雷达结构来计算点云归属于哪层线束
        if (N_SCANS == 16)
        {
            // velodyne-16 激光雷达的竖直 FoV 是[-15, -15]，分辨率是 2°，这里通过这样的计算可以对该激光点分配一个 [0, 15] 的扫描线 ID
            scanID = int((angle + 15) / 2 + 0.5); // 16线雷达为±15°视场角，+15是先把-15°补偿；/2是因为相邻scan相差2°（分辨率是 2°）；+0.5是为了四舍五入（因为int会默认去尾）
            // scanID(0~15), 这些是无效的点
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;  // 去除掉一些非法点之后的点云数量cloudSize
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            // 垂直视场角+10.67~-30.67°，垂直角度分辨率1.33°，-30.67°时的scanID = 0
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {
            // 垂直视场角+2~-24.8°，垂直角度分辨率0.4254°，+2°时的scanID = 0   
            if (angle >= -8.83)  // scanID的范围是 [0,32]
                scanID = int((2 - angle) * 3.0 + 0.5);
            else                 // scanID的范围是 [33,]
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK(); //用于中断程序并输出本句所在文件/行数
        }
        //printf("angle %f scanID %d \n", angle, scanID);
        
        // ***计算激光点水平方向角（航向角），分配时间戳***
        // 每个sweep不一定从水平0°开始，没有复位过程，雷达只在电机旋转到接近0°时记下当前对应激光点的精确坐标；
        // 同样，结束的位置，也不一定是0°，都是在0°附近，这为只使用激光点坐标计算水平角度带来一定的复杂性；
        // 若起始角度是0°，结束角度是10°+2*pi，若只通过坐标来计算水平角度，如果得到的角度是5°，那么这个激光点是开始的时候（5°）扫描的，还是结束的时候（2*pi+5°）时扫描的？
        // 所以只使用XYZ位置有时候无法得到其精确的扫描时间，还需要结合时序信息，因为一个sweep中返回的点是按照时间顺序排列的。这里变量halfPassed就是来解决这个问题的
        float ori = -atan2(point.y, point.x); 
        // 根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
        if (!halfPassed) //如果角度没有过一半圆周
        {
            // 确保-PI/2 < ori - startOri < 3/2 PI
            if (ori < startOri - M_PI / 2)
            {
                // case 1：起始点在 179°，逆时针转过几度后，当前点是 -179°，需要加上 2pi 作为补偿
                ori += 2 * M_PI;
            }
            // case 2: 理论上在逆时针转的时候不会出现这种情况，在顺时针的情况下，起始点在 -179°，
            // 顺时针转过2度后到 179°，此时差距过大，需要减去 2pi
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            
            if (ori - startOri > M_PI)
            {
                // 角度校正后如果和起始点相差超过 pi，表示已经过半圈
                halfPassed = true;
            }
        }
        else
        {
            // 经过半圈后，部分情况（扫描线从 179°到 -179°时）需要加 2pi
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                // case 1: 在逆时针下理论上不会出现
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                // case 2: 起始点在 -179°，逆时针经过半圈后当前点在 1° 
                // 此时差值是对的，因此将2pi补偿减去
                ori -= 2 * M_PI;
            }
        }
        // 角度的计算是为了计算相对起始时刻的时间，为后续点云去畸变作准备
        float relTime = (ori - startOri) / (endOri - startOri);  // relTime 是一个0~1之间的小数，代表占用扫描时间的比例，乘以扫描时间得到真实扫描时刻
        // 小技巧：用intensity的整数部分和小数部分来存储该点所属的扫描线以及相对时间：[线id].[相对时间*扫描周期]  scanPeriod扫描时间默认为0.1s
        point.intensity = scanID + scanPeriod * relTime; // 将激光雷达的线束信息和时间信息放在点云intensity的数据结构里，方便之后的程序调用
                                                         // 注意这里只是将intensity拿来记录信息用的，它跟点云的反射率等无关
                                                         // 整数部分为scanID，小数部分为时间信息（之后补偿激光点云运动畸变使用）
        //根据scan的index送入各自数组
        laserCloudScans[scanID].push_back(point); 
    }
    // cloudSize是有效的点云的数目
    cloudSize = count;  // 去除一些无效点之后的点云数量。去除了angle和SCANID不符合要求的点。
    printf("points size %d \n", cloudSize);
    


    // ***计算每一个点的曲率***
    // 将每条扫描线上的点输入到laserCloud指向的点云
    // 并记录好每条线上可以计算曲率的初始点和结束点的索引（在laserCloud中的索引）
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        // 将每个扫描线上的局部点云汇总至一个点云里面，并计算每个扫描线对应的起始和结束坐标
        // 前5个点和后5个点都无法计算曲率，因为他们不满足左右两侧各有5个点
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];  // 无序点云转换成有序点云
        scanEndInd[i] = laserCloud->size() - 6;
    }
    printf("prepare time %f \n", t_prepare.toc());  // 打印点云预处理（将一帧无序点云转换成有序点云）的耗时

    // 计算每一个点的曲率，这里的laserCloud是有序的点云，故可以直接这样计算（论文中说对每条线扫scan计算曲率）
    // 但是在每条scan的交界处计算得到的曲率是不准确的，这可通过scanStartInd[i]、scanEndInd[i]来选取
    /*
     * 表面上除了前后五个点都应该有曲率，但是由于临近点都在扫描上选取，实际上每条扫描线上的前后五个点也不太准确，
     * 应该由scanStartInd[i]、scanEndInd[i]来确定范围
     */
    for (int i = 5; i < cloudSize - 5; i++)
    { 
        // 计算当前点和周围十个点（左右各 5 个）在 x, y, z 方向上的差值： 10*p_i - sum(p_{i-5:i-1,i+1:i+5})
        // 注意这里对每一条扫描线的边缘的计算是有问题的，因为会引入相邻扫描线的点，但见上面操作，每一条扫描线我们不考虑边缘的五个点，所以为了方便可以这么操作
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 计算曲率：对应论文中的公式（1），但是没有进行除法
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ; // 三个维度的曲率平方和
        /* 
         * cloudSortInd[i] = i相当于所有点的初始自然序列，从此以后，每个点就有了它自己的序号（索引）
         * 对于每个点，我们已经选择了它附近的特征点数量初始化为0
         * 每个点的点类型初始设置为0（次极小平面点）
         */
        cloudSortInd[i] = i;  // 每个点云的index，后面根据曲率进行排序的时候使用
        cloudNeighborPicked[i] = 0;  // 这里实质上使用 1/0 来表示其相邻点是否已经选取，但是c++里面不推荐用 vector<bool> 来存储 bool
        cloudLabel[i] = 0; // 默认为0
                           // Label 2: corner_sharp
                           // Label 1: corner_less_sharp, 包含Label 2
                           // Label -1: surf_flat
                           // Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样
    }
    TicToc t_pts; //统计程序运行时间
    


    // ***按照曲率提取特征点***
    // 特征点集合用点云类保存
    pcl::PointCloud<PointType> cornerPointsSharp;       // 极大边线点
    pcl::PointCloud<PointType> cornerPointsLessSharp;   // 次极大边线点
    pcl::PointCloud<PointType> surfPointsFlat;          // 极小平面点
    pcl::PointCloud<PointType> surfPointsLessFlat;      // 次极小平面点（经过降采样）

    // 根据曲率将激光点分类为Edge Point和Planar Point
    float t_q_sort = 0;  // 用来记录排序花费的总时间
    // 遍历每个scan，按照scan的曲率顺序提取对应的4种特征点
    for (int i = 0; i < N_SCANS; i++)
    {
        // 如果最后一个可算曲率的点与第一个的差小于6，说明无法分成6个扇区，跳过
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 用来存储不太平整的点(后续对该类点进行点云降采样操作)
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);

        // 为了使特征点均匀分布，将点云均分成6块区域，每块区域选取一定数量的Edge Points，和Planar Points
        for (int j = 0; j < 6; j++) 
        {
            // 每个等分区域的起始index和结束index
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            // 将这块区域的点云按曲率从小到大排序，得到cloudSortInd，表示曲率从小到大排序的index
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // t_q_sort累计每个扇区曲率排序时间总和
            t_q_sort += t_tmp.toc();

            
            // ***下面开始挑选边点***
            // 选取极大边线点（2个）和次极大边线点（20个）
            int largestPickedNum = 0;
            // 计算Edge Points，曲率比较大的点。
            // 2个曲率最大的cornerPointsSharp，和20个cornerPointsLessSharp（这20个包含那2个cornerPointsSharp点）
            for (int k = ep; k >= sp; k--) // 从最大曲率往最小曲率遍历，寻找边线点，并要求大于0.1
            {
                // 排序后顺序就乱了，这个时候索引的作用就体现出来了
                int ind = cloudSortInd[k];  // 取出点的索引
                // 看看这个点是否是有效点，该点云的曲率必须要大于一个阈值，才能归属于Edge Points
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1)
                {
                    largestPickedNum++;
                    if (largestPickedNum <= 2) // 该subscan中曲率最大的前2个点为cornerPointsSharp特征点
                    {                        
                        cloudLabel[ind] = 2;  // label为2是曲率大的点，表示cornerPointsSharp
                        // 既放入极大边线点容器，也放入次极大边线点容器
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }

                    else if (largestPickedNum <= 20) // 该subscan中曲率最大的前20个点认为是corner_less_sharp特征点（这20个包含那2个cornerPointsSharp点）
                    {                        
                        cloudLabel[ind] = 1;  // 标签为1表示cornerPointsLessSharp特征点
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    
                    else
                    {
                        break;  // 如果已经选择的角点已经超过 20 个，则不再考虑后续的点
                    }
                    
                    // 设置相邻点选取标志位，记录这个点已经被选择了
                    cloudNeighborPicked[ind] = 1; 
                    // 为了保证特征点不过度集中，将选中的点周围5个点都打上标签，避免后续会被选到
                    // ID为ind的特征点的相邻scan点距离的平方 <= 0.05的点标记为选择过，避免特征点密集分布
                    // 右侧
                    for (int l = 1; l <= 5; l++)
                    {
                        // 如果点和前一个点的距离不超过阈值，将其标记为已经被选择中
                        // 表示这个点和某个已经被提取的特征点过近，因此不作为特征点考虑
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    // 左侧
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            
            //***下面开始挑选面点***
            // 相似的方法选取Planar Points，选取该subscan曲率最小的前4个点为surf_flat
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++) // 曲率从小到大遍历点云
            {
                int ind = cloudSortInd[k];
                //确保这个点没有被pick且曲率小于阈值
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
                {
                    //-1认为是平坦的点
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    //这里不区分平坦和比较平坦，因为剩下的点label默认是0，就是比较平坦
                    if (smallestPickedNum >= 4)
                    { 
                        break; // 如果已经选了 4 个平面点，后续的点不予考虑
                    }
                    
                    // 将其标记为已选择，并对其左右相邻 5 个距离较近的点也标记为已选择
                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 选取次极小平面点，除了极大平面点、次极大平面点，剩下的都是次极小平面点
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        
        //由于 LessFlat 的平面点数量太多（除了角点以外都是），所以这里做一个体素滤波下采样：使用体素化方法减少点云数量
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());



    // 发布去除周围的点和无效值点的原始点云
    // 发布重新组织（强度存的是：线束 + 每个点的时间）的点云
    // 发布四种特征的点云
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scam
    // 默认为false，可视化用，发布每一条scan的点云
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    //计算每次配准时间是否超过100ms（因为点云的频率为10Hz），否则会发生丢帧情况，多半是算力不够导致，一旦丢帧，后面的里程计等模块的精度就很难保证
    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    // 初始化节点 名称:scanRegistration
    // 参数1和参数2 后期为节点传值会使用
    // 参数3 是节点名称，是一个标识符，需要保证运行后，在 ROS 网络拓扑中唯一
    ros::init(argc, argv, "scanRegistration");
    /*
    * NodeHandle 是节点同ROS系统交流的主要接口
    * NodeHandle 在构造的时候会完整地初始化本节点 
    * NodeHandle 析构的时候会关闭此节点
    */
    ros::NodeHandle nh; // 实例化ros 句柄，该类封装了 ROS 中的一些常用功能
    
    //从launch配置参数中 读取 scan_line 参数, 多少线的激光雷达  在launch文件中配置的
    nh.param<int>("scan_line", N_SCANS, 16);
    //从launch配置参数中 读取 minimum_range 参数, 最小有效距离（剔除太近的点不让在地图中显示，因为这些点可能是激光雷达的载体会影响视野） 
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1); // 剔除当前LiDAR坐标系下离原点0.1m以内的点，0.1只是默认值，16线的launch中是0.3

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
   
    //订阅 velodyne_cloud 的消息 收到一个topic消息包,则执行 laserCloudHandler 回调函数一次
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    //以下是要发布出去的topic，由程序员决定什么时候发布，发布就直接调用publish即可
    //定义：发布处理后的 所有点云 角点特征 弱角点特征 平面特征 弱平面特征 界外(去除)点
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);  // 每100ms（0.1s）发布一次
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    // 发布每条线束 默认不发送
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }

    ros::spin();  // 保证你指定的回调函数会被调用,程序执行到spin()后就不调用其他语句了
    return 0;
}

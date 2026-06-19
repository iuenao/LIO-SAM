#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
#include "lio_sam/srv/save_map.hpp"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

struct ResidualReliability
{
    float geometry = 1.0f;
    float residual = 1.0f;
    float weight = 1.0f;
    float rawResidual = 0.0f;
    float scaledResidual = 0.0f;
    float lioSamBaseScale = 1.0f;
    int featureType = 0; // 1: corner, 2: surf
};

struct FinalCorrespondence
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class FeatureType : uint8_t
    {
        Corner = 1,
        Surface = 2
    };

    FeatureType type = FeatureType::Surface;
    double rawResidual = 0.0;
    double lioSamBaseScale = 1.0;
    Eigen::Matrix<double, 1, 6> baseJacobian = Eigen::Matrix<double, 1, 6>::Zero();
    double geometryReliability = 1.0;
    double residualReliability = 1.0;
};

struct FactorInformationDiagnostics
{
    double timestamp = 0.0;
    int registrationWeightMode = 0;
    int factorCovarianceMode = 0;
    int factorCovarianceScaleMode = 0;
    double factorCovarianceAdaptiveBlend = 1.0;
    int numCorner = 0;
    int numSurface = 0;
    int numTotal = 0;
    double meanRawResidual = 0.0;
    double rmseRawResidual = 0.0;
    double meanGeometryReliability = 1.0;
    double meanResidualReliability = 1.0;
    double meanCombinedReliability = 1.0;
    double effectiveCorrespondenceRatio = 1.0;
    Eigen::Matrix<double, 6, 1> rawInformationEigenvalues = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> weightedInformationEigenvalues = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> covarianceDiagonal = Eigen::Matrix<double, 6, 1>::Zero();
    double covarianceTrace = 0.0;
    double covarianceRotationTrace = 0.0;
    double covarianceTranslationTrace = 0.0;
    double rawConditionNumber = 0.0;
    double weightedConditionNumber = 0.0;
    bool covarianceValid = false;
    bool usedFallback = false;
    bool isDegenerate = false;
    int lmIterations = 0;
    double registrationRuntimeMs = 0.0;
    double finalInformationRuntimeMs = 0.0;
    double transformUpdateDeltaRotationDeg = 0.0;
    double transformUpdateDeltaTranslationM = 0.0;
};

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;

    rclcpp::Service<lio_sam::srv::SaveMap>::SharedPtr srvSaveMap;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;

    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    lio_sam::msg::CloudInfo cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    std::vector<ResidualReliability> reliabilitySel;
    bool finalCorrespondencesValid = false;
    std::vector<FinalCorrespondence, Eigen::aligned_allocator<FinalCorrespondence>> finalCorrespondences;
    bool lastLidarFactorCovarianceValid = false;
    Eigen::Matrix<double, 6, 6> lastLidarFactorCovarianceLoam = Eigen::Matrix<double, 6, 6>::Identity();
    gtsam::Matrix6 lastLidarFactorCovarianceGtsam = gtsam::Matrix6::Identity();
    double lastLidarFactorCovarianceTimestamp = -1.0;
    int lastLidarFactorSourceKeyframe = -1;
    Eigen::Matrix<double, 6, 6> lastRawInformation = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> lastWeightedInformation = Eigen::Matrix<double, 6, 6>::Zero();
    FactorInformationDiagnostics lastFactorInformationDiagnostics;
    bool covarianceConversionSelfTestPassed = false;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<ResidualReliability> reliabilityCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<ResidualReliability> reliabilitySurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    std::ofstream trajectoryFile;
    std::ofstream tumTrajectoryFile;
    std::ofstream reliabilityDiagnosticsFile;
    std::ofstream scanDiagnosticsFile;
    std::ofstream keyframeDiagnosticsFile;
    bool trajectoryFileOpen = false;
    bool tumTrajectoryFileOpen = false;
    bool reliabilityDiagnosticsFileOpen = false;
    bool scanDiagnosticsFileOpen = false;
    bool keyframeDiagnosticsFileOpen = false;
    bool enableTrajectoryCSV = false;
    bool enableDiagnosticsCSV = false;
    bool enablePointLevelReliabilityCSV = false;
    std::string diagnosticsOutputDir;
    std::string diagnosticsFilePrefix;
    std::string runId;
    std::string datasetName;
    std::string sequenceName;
    std::string methodName;
    std::string configFile;
    std::string bagName;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    std::unique_ptr<tf2_ros::TransformBroadcaster> br;

    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options)
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry", qos);
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos);
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        subCloud = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        subGPS = create_subscription<nav_msgs::msg::Odometry>(
            gpsTopic, 200,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));
        subLoop = create_subscription<std_msgs::msg::Float64MultiArray>(
            "lio_loop/loop_closure_detection", qos,
            std::bind(&mapOptimization::loopInfoHandler, this, std::placeholders::_1));

        auto saveMapService = [this](const std::shared_ptr<rmw_request_id_t> request_header, const std::shared_ptr<lio_sam::srv::SaveMap::Request> req, std::shared_ptr<lio_sam::srv::SaveMap::Response> res) -> void {
            (void)request_header;
            string saveMapDirectory;
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files ..." << endl;
            if(req->destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
            else saveMapDirectory = std::getenv("HOME") + req->destination;
            cout << "Save destination: " << saveMapDirectory << endl;
            // create directory and remove old files;
            int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
            unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
            // save key frame transformations
            pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
            // extract global point cloud map
            pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
            {
                *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
                *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
                cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
            }
            if(req->resolution != 0)
            {
               cout << "\n\nSave resolution: " << req->resolution << endl;
               // down-sample and save corner cloud
               downSizeFilterCorner.setInputCloud(globalCornerCloud);
               downSizeFilterCorner.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterCorner.filter(*globalCornerCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
               // down-sample and save surf cloud
               downSizeFilterSurf.setInputCloud(globalSurfCloud);
               downSizeFilterSurf.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterSurf.filter(*globalSurfCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
            }
            else
            {
            // save corner cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
               // save surf cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
            }
            // save global point cloud map
            *globalMapCloud += *globalCornerCloud;
            *globalMapCloud += *globalSurfCloud;
            int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
            res->success = ret == 0;
            downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
            downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files completed\n" << endl;
            return;
        };
        
        srvSaveMap = create_service<lio_sam::srv::SaveMap>("lio_sam/save_map", saveMapService);
        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();

        covarianceConversionSelfTestPassed = runCovarianceConversionSelfTest();

        initialiseLoggers();
    }

    void initialiseLoggers()
    {
        enableTrajectoryCSV = declare_parameter<bool>("enableTrajectoryCSV", false);
        enableDiagnosticsCSV = declare_parameter<bool>("enableDiagnosticsCSV", false);
        enablePointLevelReliabilityCSV = declare_parameter<bool>("enablePointLevelReliabilityCSV", false);
        diagnosticsOutputDir = declare_parameter<std::string>(
            "diagnosticsOutputDir", "/home/unitree/ros2_workspaces/lio_ws/lio_sam_logs");
        diagnosticsFilePrefix = declare_parameter<std::string>("diagnosticsFilePrefix", "run");
        runId = declare_parameter<std::string>("run_id", diagnosticsFilePrefix);
        datasetName = declare_parameter<std::string>("dataset_name", "unknown");
        sequenceName = declare_parameter<std::string>("sequence_name", "unknown");
        methodName = declare_parameter<std::string>("method_name", "fixed");
        configFile = declare_parameter<std::string>("config_file", "unknown");
        bagName = declare_parameter<std::string>("bag_name", "unknown");

        if (!enableTrajectoryCSV && !enableDiagnosticsCSV)
            return;

        std::error_code ec;
        std::filesystem::create_directories(diagnosticsOutputDir, ec);
        if (ec)
        {
            RCLCPP_WARN(get_logger(),
                "Could not create diagnostics directory %s: %s",
                diagnosticsOutputDir.c_str(), ec.message().c_str());
            return;
        }

        if (enableDiagnosticsCSV)
            saveConfigSnapshot();

        if (enableTrajectoryCSV)
        {
            std::filesystem::path trajectoryPath =
                std::filesystem::path(diagnosticsOutputDir) / (diagnosticsFilePrefix + "_trajectory.csv");
            trajectoryFile.open(trajectoryPath, std::ios::out | std::ios::trunc);
            if (trajectoryFile.is_open())
            {
                trajectoryFileOpen = true;
                trajectoryFile
                    << "timestamp,x,y,z,qx,qy,qz,qw,roll,pitch,yaw,"
                    << "run_id,dataset_name,sequence_name,method_name,config_file,bag_name"
                    << '\n';
                RCLCPP_INFO(get_logger(), "Trajectory CSV opened at %s", trajectoryPath.string().c_str());
            }
            else
            {
                RCLCPP_WARN(get_logger(), "Failed to open trajectory CSV at %s", trajectoryPath.string().c_str());
            }

            std::filesystem::path tumPath =
                std::filesystem::path(diagnosticsOutputDir) / (diagnosticsFilePrefix + "_tum.txt");
            tumTrajectoryFile.open(tumPath, std::ios::out | std::ios::trunc);
            if (tumTrajectoryFile.is_open())
            {
                tumTrajectoryFileOpen = true;
                RCLCPP_INFO(get_logger(), "TUM trajectory opened at %s", tumPath.string().c_str());
            }
            else
            {
                RCLCPP_WARN(get_logger(), "Failed to open TUM trajectory at %s", tumPath.string().c_str());
            }
        }

        if (enableDiagnosticsCSV)
        {
            std::filesystem::path scanPath =
                std::filesystem::path(diagnosticsOutputDir) / (diagnosticsFilePrefix + "_scan_diagnostics.csv");
            scanDiagnosticsFile.open(scanPath, std::ios::out | std::ios::trunc);
            if (scanDiagnosticsFile.is_open())
            {
                scanDiagnosticsFileOpen = true;
                writeAggregateDiagnosticsHeader(scanDiagnosticsFile);
                RCLCPP_INFO(get_logger(), "Scan diagnostics CSV opened at %s", scanPath.string().c_str());
            }
            else
            {
                RCLCPP_WARN(get_logger(), "Failed to open scan diagnostics CSV at %s", scanPath.string().c_str());
            }

            std::filesystem::path keyframePath =
                std::filesystem::path(diagnosticsOutputDir) / (diagnosticsFilePrefix + "_keyframe_factors.csv");
            keyframeDiagnosticsFile.open(keyframePath, std::ios::out | std::ios::trunc);
            if (keyframeDiagnosticsFile.is_open())
            {
                keyframeDiagnosticsFileOpen = true;
                writeAggregateDiagnosticsHeader(keyframeDiagnosticsFile);
                RCLCPP_INFO(get_logger(), "Keyframe factor diagnostics CSV opened at %s", keyframePath.string().c_str());
            }
            else
            {
                RCLCPP_WARN(get_logger(), "Failed to open keyframe factor diagnostics CSV at %s", keyframePath.string().c_str());
            }

            if (enablePointLevelReliabilityCSV)
            {
                std::filesystem::path reliabilityPath =
                    std::filesystem::path(diagnosticsOutputDir) / (diagnosticsFilePrefix + "_reliability.csv");
                reliabilityDiagnosticsFile.open(reliabilityPath, std::ios::out | std::ios::trunc);
                if (reliabilityDiagnosticsFile.is_open())
                {
                    reliabilityDiagnosticsFileOpen = true;
                    reliabilityDiagnosticsFile
                        << "timestamp,iteration,index,feature_type,raw_residual,registration_weight,"
                        << "geometry_reliability,residual_reliability,legacy_reliability_weight,total_solve_weight"
                        << '\n';
                    RCLCPP_INFO(get_logger(), "Point-level reliability CSV opened at %s", reliabilityPath.string().c_str());
                }
                else
                {
                    RCLCPP_WARN(get_logger(), "Failed to open point-level reliability CSV at %s", reliabilityPath.string().c_str());
                }
            }
        }
    }

    void writeAggregateDiagnosticsHeader(std::ofstream& file)
    {
        file
            << "timestamp,keyframe_index,registration_weight_mode,factor_covariance_mode,"
            << "factor_covariance_scale_mode,factor_covariance_adaptive_blend,"
            << "num_corner,num_surface,num_total,"
            << "mean_abs_raw_residual,rmse_raw_residual,"
            << "mean_geometry_reliability,mean_residual_reliability,mean_combined_reliability,"
            << "effective_correspondence_ratio,"
            << "raw_info_eig_0,raw_info_eig_1,raw_info_eig_2,raw_info_eig_3,raw_info_eig_4,raw_info_eig_5,"
            << "weighted_info_eig_0,weighted_info_eig_1,weighted_info_eig_2,weighted_info_eig_3,weighted_info_eig_4,weighted_info_eig_5,"
            << "raw_condition_number,weighted_condition_number,"
            << "cov_0_0,cov_1_1,cov_2_2,cov_3_3,cov_4_4,cov_5_5,"
            << "covariance_trace,covariance_rotation_trace,covariance_translation_trace,"
            << "covariance_valid,used_fixed_fallback,is_degenerate,"
            << "lm_iterations,registration_runtime_ms,factor_information_runtime_ms,"
            << "transform_update_delta_rotation_deg,transform_update_delta_translation_m"
            << '\n';
    }

    void writeAggregateDiagnosticsRow(
        std::ofstream& file,
        int keyframeIndex,
        const FactorInformationDiagnostics& diagnostics,
        bool usedFixedFallback)
    {
        file << std::fixed << std::setprecision(9)
             << diagnostics.timestamp << ","
             << keyframeIndex << ","
             << diagnostics.registrationWeightMode << ","
             << diagnostics.factorCovarianceMode << ","
             << diagnostics.factorCovarianceScaleMode << ","
             << diagnostics.factorCovarianceAdaptiveBlend << ","
             << diagnostics.numCorner << ","
             << diagnostics.numSurface << ","
             << diagnostics.numTotal << ","
             << diagnostics.meanRawResidual << ","
             << diagnostics.rmseRawResidual << ","
             << diagnostics.meanGeometryReliability << ","
             << diagnostics.meanResidualReliability << ","
             << diagnostics.meanCombinedReliability << ","
             << diagnostics.effectiveCorrespondenceRatio;

        for (int i = 0; i < 6; ++i)
            file << "," << diagnostics.rawInformationEigenvalues(i);
        for (int i = 0; i < 6; ++i)
            file << "," << diagnostics.weightedInformationEigenvalues(i);
        file << ","
             << diagnostics.rawConditionNumber << ","
             << diagnostics.weightedConditionNumber;
        for (int i = 0; i < 6; ++i)
            file << "," << diagnostics.covarianceDiagonal(i);
        file << ","
             << diagnostics.covarianceTrace << ","
             << diagnostics.covarianceRotationTrace << ","
             << diagnostics.covarianceTranslationTrace << ","
             << (diagnostics.covarianceValid ? 1 : 0) << ","
             << (usedFixedFallback ? 1 : 0) << ","
             << (diagnostics.isDegenerate ? 1 : 0) << ","
             << diagnostics.lmIterations << ","
             << diagnostics.registrationRuntimeMs << ","
             << diagnostics.finalInformationRuntimeMs << ","
             << diagnostics.transformUpdateDeltaRotationDeg << ","
             << diagnostics.transformUpdateDeltaTranslationM
             << '\n';
    }

    void writeScanDiagnostics(int keyframeIndex, const FactorInformationDiagnostics& diagnostics)
    {
        if (scanDiagnosticsFileOpen)
            writeAggregateDiagnosticsRow(scanDiagnosticsFile, keyframeIndex, diagnostics, diagnostics.usedFallback);
    }

    void writeKeyframeFactorDiagnostics(
        int keyframeIndex,
        const FactorInformationDiagnostics& diagnostics,
        bool usedFixedFallback)
    {
        if (keyframeDiagnosticsFileOpen)
            writeAggregateDiagnosticsRow(keyframeDiagnosticsFile, keyframeIndex, diagnostics, usedFixedFallback);
    }

    std::string csvEscape(const std::string& value) const
    {
        if (value.find_first_of(",\"\n") == std::string::npos)
            return value;

        std::string escaped = "\"";
        for (char c : value)
        {
            if (c == '"')
                escaped += "\"\"";
            else
                escaped += c;
        }
        escaped += "\"";
        return escaped;
    }

    void saveConfigSnapshot()
    {
        std::filesystem::path snapshotPath =
            std::filesystem::path(diagnosticsOutputDir) / (diagnosticsFilePrefix + "_config_snapshot.yaml");
        std::ofstream snapshot(snapshotPath, std::ios::out | std::ios::trunc);
        if (!snapshot.is_open())
        {
            RCLCPP_WARN(get_logger(), "Failed to write config snapshot at %s", snapshotPath.string().c_str());
            return;
        }

        snapshot << "# Runtime parameter snapshot generated by lio_sam_mapOptimization\n";
        for (const auto& name : list_parameters({}, 10).names)
        {
            rclcpp::Parameter parameter;
            if (get_parameter(name, parameter))
                snapshot << name << ": " << parameter.value_to_string() << "\n";
        }
    }

    void writeTrajectoryLogs(const PointTypePose& pose_in, const tf2::Quaternion& q)
    {
        if (!trajectoryFileOpen && !tumTrajectoryFileOpen)
            return;

        if (trajectoryFileOpen)
        {
            trajectoryFile << std::fixed << std::setprecision(9)
                           << pose_in.time << ","
                           << pose_in.x << "," << pose_in.y << "," << pose_in.z << ","
                           << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << ","
                           << pose_in.roll << "," << pose_in.pitch << "," << pose_in.yaw << ","
                           << csvEscape(runId) << ","
                           << csvEscape(datasetName) << ","
                           << csvEscape(sequenceName) << ","
                           << csvEscape(methodName) << ","
                           << csvEscape(configFile) << ","
                           << csvEscape(bagName)
                           << '\n';
        }

        if (tumTrajectoryFileOpen)
        {
            tumTrajectoryFile << std::fixed << std::setprecision(9)
                              << pose_in.time << " "
                              << pose_in.x << " " << pose_in.y << " " << pose_in.z << " "
                              << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                              << '\n';
        }
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        reliabilityCornerVec.resize(N_SCAN * Horizon_SCAN);
        reliabilitySurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void visualizeGlobalMapThread()
    {
        rclcpp::Rate rate(0.2);
        while (rclcpp::ok()){
            rate.sleep();
            publishGlobalMap();
        }
        if (savePCD == false)
            return;
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround->get_subscription_count() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }












    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        rclcpp::Rate rate(loopClosureFrequency);
        while (rclcpp::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::msg::Float64MultiArray::SharedPtr loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames->get_subscription_count() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::msg::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::msg::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::msg::Marker::ADD;
        markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::msg::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::msg::Marker::ADD;
        markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::msg::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge->publish(markerArray);
    }







    



    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odom_available == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(
                cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
                cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imu_available == true && imuType)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    float clampReliability(float value) const
    {
        const float minValue = std::max(reliabilityMin, 1e-3f);
        return std::clamp(value, minValue, 1.0f);
    }

    float computeCornerGeometryReliability(float lambda0, float lambda1) const
    {
        const float ratio = lambda0 / std::max(lambda1, 1e-6f);
        const float threshold = 3.0f;
        const float reference = std::max(reliabilityCornerRatioRef, threshold + 1e-3f);
        return clampReliability((ratio - threshold) / (reference - threshold));
    }

    float computeSurfGeometryReliability(float maxPlaneDistance) const
    {
        const float scale = std::max(reliabilitySurfFitScale, 1e-6f);
        return clampReliability(std::exp(-(maxPlaneDistance * maxPlaneDistance) / (scale * scale)));
    }

    float computeResidualReliability(float residual) const
    {
        const float scale = std::max(reliabilityResidualScale, 1e-6f);
        return clampReliability(std::exp(-(residual * residual) / (scale * scale)));
    }

    ResidualReliability makeReliability(
        float geometryReliability,
        float rawResidual,
        float scaledResidual,
        float lioSamBaseScale,
        int featureType) const
    {
        ResidualReliability reliability;
        reliability.geometry = clampReliability(geometryReliability);
        reliability.residual = computeResidualReliability(rawResidual);
        reliability.rawResidual = rawResidual;
        reliability.scaledResidual = scaledResidual;
        reliability.lioSamBaseScale = lioSamBaseScale;
        reliability.featureType = featureType;

        if (!adaptiveCovEnabled)
        {
            reliability.weight = 1.0f;
            return reliability;
        }

        switch (adaptiveCovMode)
        {
            case 1:
                reliability.weight = reliability.geometry;
                break;
            case 2:
                reliability.weight = reliability.residual;
                break;
            case 3:
                reliability.weight = reliability.geometry * reliability.residual;
                break;
            default:
                reliability.weight = 1.0f;
                break;
        }

        reliability.weight = std::clamp(
            reliability.weight,
            std::max(reliabilityWeightMin, 1e-6f),
            std::max(reliabilityWeightMax, std::max(reliabilityWeightMin, 1e-6f)));
        return reliability;
    }

    float computeRegistrationWeight(float rawResidual, float geometryReliability) const
    {
        const float geom = std::clamp(
            geometryReliability,
            std::max(reliabilityWeightMin, 1e-6f),
            std::max(reliabilityWeightMax, std::max(reliabilityWeightMin, 1e-6f)));

        switch (registrationWeightMode)
        {
            case 1:
                return computeHuberWeight(rawResidual);
            case 2:
                return computeCauchyWeight(rawResidual);
            case 3:
                return geom * computeHuberWeight(rawResidual);
            case 4:
                return geom;
            case 5:
                return computeResidualReliability(rawResidual);
            case 6:
                return geom * computeResidualReliability(rawResidual);
            default:
                return 1.0f;
        }
    }

    Eigen::Matrix<double, 1, 6> computeBaseJacobianRow(const PointType& pointOriLidar, const PointType& coeffLidar) const
    {
        const double srx = sin(transformTobeMapped[1]);
        const double crx = cos(transformTobeMapped[1]);
        const double sry = sin(transformTobeMapped[2]);
        const double cry = cos(transformTobeMapped[2]);
        const double srz = sin(transformTobeMapped[0]);
        const double crz = cos(transformTobeMapped[0]);

        PointType pointOri, coeff;
        pointOri.x = pointOriLidar.y;
        pointOri.y = pointOriLidar.z;
        pointOri.z = pointOriLidar.x;
        coeff.x = coeffLidar.y;
        coeff.y = coeffLidar.z;
        coeff.z = coeffLidar.x;

        const double arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                         + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                         + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

        const double ary = ((cry*srx*srz - crz*sry)*pointOri.x
                         + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                         + ((-cry*crz - srx*sry*srz)*pointOri.x
                         + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

        const double arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                         + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                         + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

        Eigen::Matrix<double, 1, 6> row;
        // Same LOAM/LIO-SAM row ordering as matA; includes the original LIO-SAM scale s through coeffLidar.
        row << arz, arx, ary, coeff.z, coeff.x, coeff.y;
        return row;
    }

    void addFinalCorrespondence(const PointType& pointOri, const PointType& coeff, const ResidualReliability& reliability)
    {
        FinalCorrespondence correspondence;
        correspondence.type = reliability.featureType == 1
            ? FinalCorrespondence::FeatureType::Corner
            : FinalCorrespondence::FeatureType::Surface;
        correspondence.rawResidual = reliability.rawResidual;
        correspondence.lioSamBaseScale = reliability.lioSamBaseScale;
        correspondence.baseJacobian = computeBaseJacobianRow(pointOri, coeff);
        correspondence.geometryReliability = reliability.geometry;
        correspondence.residualReliability = reliability.residual;
        finalCorrespondences.push_back(correspondence);
    }

    bool matrixFinite(const Eigen::Matrix<double, 6, 6>& matrix) const
    {
        return matrix.allFinite();
    }

    Eigen::Matrix<double, 6, 6> fixedOdometryCovarianceLoam() const
    {
        Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
        covariance.diagonal() << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        return covariance;
    }

    gtsam::Pose3 lmArrayToGtsamPose(const std::array<float, 6>& transformIn) const
    {
        return gtsam::Pose3(
            gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    std::array<float, 6> currentLmPoseArray() const
    {
        std::array<float, 6> pose{};
        for (int i = 0; i < 6; ++i)
            pose[i] = transformTobeMapped[i];
        return pose;
    }

    bool matrixPositiveDefinite(const Eigen::Matrix<double, 6, 6>& matrix) const
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(0.5 * (matrix + matrix.transpose()));
        return solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() > 0.0;
    }

    bool computeLmToRelativeFactorJacobian(
        const gtsam::Pose3& previousPose,
        const std::array<float, 6>& currentLmPose,
        Eigen::Matrix<double, 6, 6>* jacobian,
        std::string* errorMessage) const
    {
        if (!jacobian)
        {
            if (errorMessage)
                *errorMessage = "null Jacobian output";
            return false;
        }

        const double rotationEpsilon = std::max(covarianceJacobianRotationEpsilon, 1e-9);
        const double translationEpsilon = std::max(covarianceJacobianTranslationEpsilon, 1e-9);
        const gtsam::Pose3 currentPose = lmArrayToGtsamPose(currentLmPose);
        const gtsam::Pose3 nominalRelative = previousPose.between(currentPose);
        jacobian->setZero();

        for (int column = 0; column < 6; ++column)
        {
            const double epsilon = column < 3 ? rotationEpsilon : translationEpsilon;
            std::array<float, 6> plusPose = currentLmPose;
            std::array<float, 6> minusPose = currentLmPose;
            plusPose[column] += epsilon;
            minusPose[column] -= epsilon;

            const gtsam::Pose3 plusRelative = previousPose.between(lmArrayToGtsamPose(plusPose));
            const gtsam::Pose3 minusRelative = previousPose.between(lmArrayToGtsamPose(minusPose));
            const gtsam::Vector6 xiPlus =
                gtsam::Pose3::Logmap(nominalRelative.inverse().compose(plusRelative));
            const gtsam::Vector6 xiMinus =
                gtsam::Pose3::Logmap(nominalRelative.inverse().compose(minusRelative));

            jacobian->col(column) = (xiPlus - xiMinus) / (2.0 * epsilon);
        }

        if (!matrixFinite(*jacobian))
        {
            if (errorMessage)
                *errorMessage = "non-finite LM-to-relative tangent Jacobian";
            jacobian->setZero();
            return false;
        }

        return true;
    }

    bool convertLmPoseCovarianceToRelativeFactorCovariance(
        const gtsam::Pose3& previousPose,
        const std::array<float, 6>& currentLmPose,
        const Eigen::Matrix<double, 6, 6>& covarianceLm,
        gtsam::Matrix6* covarianceRelative,
        std::string* errorMessage) const
    {
        if (!covarianceRelative)
        {
            if (errorMessage)
                *errorMessage = "null relative covariance output";
            return false;
        }

        if (!matrixFinite(covarianceLm) || !matrixPositiveDefinite(covarianceLm))
        {
            if (errorMessage)
                *errorMessage = "invalid LM covariance";
            return false;
        }

        Eigen::Matrix<double, 6, 6> jacobian;
        if (!computeLmToRelativeFactorJacobian(previousPose, currentLmPose, &jacobian, errorMessage))
            return false;

        Eigen::Matrix<double, 6, 6> covariance =
            jacobian * covarianceLm * jacobian.transpose();
        covariance = 0.5 * (covariance + covariance.transpose());

        if (!matrixFinite(covariance))
        {
            if (errorMessage)
                *errorMessage = "non-finite relative factor covariance";
            return false;
        }

        if (!matrixPositiveDefinite(covariance))
        {
            if (errorMessage)
                *errorMessage = "non-positive-definite relative factor covariance";
            return false;
        }

        *covarianceRelative = covariance;
        return true;
    }

    void setFixedLidarFactorCovarianceFallback()
    {
        lastLidarFactorCovarianceLoam = fixedOdometryCovarianceLoam();
        lastLidarFactorCovarianceGtsam = lastLidarFactorCovarianceLoam;
    }

    void invalidateLidarFactorCovariance()
    {
        lastLidarFactorCovarianceValid = false;
        lastLidarFactorCovarianceTimestamp = -1.0;
        lastLidarFactorSourceKeyframe = -1;
    }

    bool runCovarianceConversionSelfTest()
    {
        Eigen::Matrix<double, 6, 6> covarianceLoam = Eigen::Matrix<double, 6, 6>::Zero();
        covarianceLoam.diagonal() << 1e-8, 2e-8, 3e-8, 1e-6, 2e-6, 3e-6;

        const std::array<float, 6> zeroPose{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const std::array<float, 6> nonzeroPose{0.05f, -0.04f, 0.08f, 1.0f, -0.5f, 0.2f};
        const gtsam::Pose3 identityPose;
        const gtsam::Pose3 previousPose(
            gtsam::Rot3::RzRyRx(-0.03, 0.02, -0.05),
            gtsam::Point3(-0.3, 0.7, 0.1));

        gtsam::Matrix6 covarianceZero = gtsam::Matrix6::Identity();
        gtsam::Matrix6 covarianceNonzero = gtsam::Matrix6::Identity();
        std::string errorMessage;
        const bool passed =
            convertLmPoseCovarianceToRelativeFactorCovariance(
                identityPose, zeroPose, covarianceLoam, &covarianceZero, &errorMessage) &&
            convertLmPoseCovarianceToRelativeFactorCovariance(
                previousPose, nonzeroPose, covarianceLoam, &covarianceNonzero, &errorMessage);

        if (passed)
        {
            RCLCPP_INFO(
                get_logger(),
                "Covariance conversion self-test passed: finite-difference LM pose covariance -> relative Pose3 tangent covariance.");
        }
        else
        {
            RCLCPP_ERROR(
                get_logger(),
                "Covariance conversion self-test failed (%s); adaptive LiDAR factor covariance will fall back to fixed covariance.",
                errorMessage.c_str());
        }

        return passed;
    }

    double computeFactorWeight(const FinalCorrespondence& correspondence) const
    {
        double weight = 1.0;

        switch (factorCovarianceMode)
        {
            case 1:
                weight = 1.0;
                break;
            case 2:
                weight = correspondence.residualReliability;
                break;
            case 3:
                weight = correspondence.geometryReliability;
                break;
            case 4:
                weight = correspondence.geometryReliability * correspondence.residualReliability;
                break;
            case 5:
                weight = computeRegistrationWeight(correspondence.rawResidual, correspondence.geometryReliability);
                break;
            default:
                weight = 1.0;
                break;
        }

        return std::clamp(
            weight,
            std::max((double)reliabilityWeightMin, 1e-12),
            std::max((double)reliabilityWeightMax, std::max((double)reliabilityWeightMin, 1e-12)));
    }

    double conditionNumberFromEigenvalues(const Eigen::Matrix<double, 6, 1>& eigenvalues) const
    {
        const double minEigen = std::max(eigenvalues.minCoeff(), 1e-12);
        const double maxEigen = std::max(eigenvalues.maxCoeff(), minEigen);
        return maxEigen / minEigen;
    }

    Eigen::Matrix<double, 6, 1> sortedSelfAdjointEigenvalues(const Eigen::Matrix<double, 6, 6>& matrix) const
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(0.5 * (matrix + matrix.transpose()));
        if (solver.info() != Eigen::Success)
            return Eigen::Matrix<double, 6, 1>::Zero();
        return solver.eigenvalues();
    }

    Eigen::Matrix<double, 6, 6> blockTraceNormalizeToFixedCovariance(
        const Eigen::Matrix<double, 6, 6>& covariance) const
    {
        const Eigen::Matrix<double, 6, 6> fixedCovariance = fixedOdometryCovarianceLoam();
        const double epsilon = 1e-18;
        const double rotationTrace = std::max(covariance.block<3, 3>(0, 0).trace(), epsilon);
        const double translationTrace = std::max(covariance.block<3, 3>(3, 3).trace(), epsilon);
        const double fixedRotationTrace = fixedCovariance.block<3, 3>(0, 0).trace();
        const double fixedTranslationTrace = fixedCovariance.block<3, 3>(3, 3).trace();

        const double rotationScale = std::sqrt(fixedRotationTrace / rotationTrace);
        const double translationScale = std::sqrt(fixedTranslationTrace / translationTrace);

        Eigen::Matrix<double, 6, 6> scale = Eigen::Matrix<double, 6, 6>::Identity();
        for (int i = 0; i < 3; ++i)
            scale(i, i) = rotationScale;
        for (int i = 3; i < 6; ++i)
            scale(i, i) = translationScale;

        Eigen::Matrix<double, 6, 6> normalized = scale * covariance * scale;
        return 0.5 * (normalized + normalized.transpose());
    }

    double normalizedAngleDifference(double angle) const
    {
        return std::remainder(angle, 2.0 * M_PI);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        reliabilityCornerVec[i] = makeReliability(
                            computeCornerGeometryReliability(matD1.at<float>(0, 0), matD1.at<float>(0, 1)),
                            ld2,
                            coeff.intensity,
                            s,
                            1);
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                float maxPlaneDistance = 0.0f;
                for (int j = 0; j < 5; j++) {
                    const float planeDistance = fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                                      pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                                      pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd);
                    maxPlaneDistance = std::max(maxPlaneDistance, planeDistance);
                    if (planeDistance > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        reliabilitySurfVec[i] = makeReliability(
                            computeSurfGeometryReliability(maxPlaneDistance),
                            pd2,
                            coeff.intensity,
                            s,
                            2);
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
                reliabilitySel.push_back(reliabilityCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
                reliabilitySel.push_back(reliabilitySurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool computeFinalCorrespondencesAtConvergedPose()
    {
        updatePointAssociateToMap();
        finalCorrespondences.clear();
        finalCorrespondences.reserve(laserCloudCornerLastDSNum + laserCloudSurfLastDSNum);

        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            if (kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis) < 5)
                continue;

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5; cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;
                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        ResidualReliability reliability = makeReliability(
                            computeCornerGeometryReliability(matD1.at<float>(0, 0), matD1.at<float>(0, 1)),
                            ld2,
                            coeff.intensity,
                            s,
                            1);
                        addFinalCorrespondence(pointOri, coeff, reliability);
                    }
                }
            }
        }

        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            if (kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis) < 5)
                continue;

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                float maxPlaneDistance = 0.0f;
                for (int j = 0; j < 5; j++) {
                    const float planeDistance = fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                                      pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                                      pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd);
                    maxPlaneDistance = std::max(maxPlaneDistance, planeDistance);
                    if (planeDistance > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        ResidualReliability reliability = makeReliability(
                            computeSurfGeometryReliability(maxPlaneDistance),
                            pd2,
                            coeff.intensity,
                            s,
                            2);
                        addFinalCorrespondence(pointOri, coeff, reliability);
                    }
                }
            }
        }

        return (int)finalCorrespondences.size() >= factorCovarianceMinCorrespondences;
    }

    bool computeLidarFactorInformationAndCovariance(FactorInformationDiagnostics* diagnostics)
    {
        invalidateLidarFactorCovariance();
        lastRawInformation.setZero();
        lastWeightedInformation.setZero();

        FactorInformationDiagnostics localDiagnostics;
        localDiagnostics.timestamp = timeLaserInfoCur;
        localDiagnostics.registrationWeightMode = registrationWeightMode;
        localDiagnostics.factorCovarianceMode = factorCovarianceMode;
        localDiagnostics.factorCovarianceScaleMode = factorCovarianceScaleMode;
        localDiagnostics.factorCovarianceAdaptiveBlend = factorCovarianceAdaptiveBlend;

        if (factorCovarianceMode == 0)
        {
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        if (!covarianceConversionSelfTestPassed)
        {
            localDiagnostics.usedFallback = factorCovarianceFallbackToFixed;
            if (factorCovarianceFallbackToFixed)
                setFixedLidarFactorCovarianceFallback();
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        if (!finalCorrespondencesValid || (int)finalCorrespondences.size() < factorCovarianceMinCorrespondences)
        {
            localDiagnostics.numTotal = finalCorrespondences.size();
            localDiagnostics.usedFallback = factorCovarianceFallbackToFixed;
            if (factorCovarianceFallbackToFixed)
                setFixedLidarFactorCovarianceFallback();
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        double absResidualSum = 0.0;
        double residualSquareSum = 0.0;
        double geometryReliabilitySum = 0.0;
        double residualReliabilitySum = 0.0;
        double combinedReliabilitySum = 0.0;
        double selectedWeightSum = 0.0;

        for (const auto& correspondence : finalCorrespondences)
        {
            if (correspondence.type == FinalCorrespondence::FeatureType::Corner)
                localDiagnostics.numCorner++;
            else
                localDiagnostics.numSurface++;

            const Eigen::Matrix<double, 6, 1> jacobianT = correspondence.baseJacobian.transpose();
            const Eigen::Matrix<double, 6, 6> contribution = jacobianT * correspondence.baseJacobian;
            const double factorWeight = computeFactorWeight(correspondence);

            lastRawInformation += contribution;
            lastWeightedInformation += factorWeight * contribution;

            absResidualSum += std::abs(correspondence.rawResidual);
            residualSquareSum += correspondence.rawResidual * correspondence.rawResidual;
            geometryReliabilitySum += correspondence.geometryReliability;
            residualReliabilitySum += correspondence.residualReliability;
            combinedReliabilitySum += correspondence.geometryReliability * correspondence.residualReliability;
            selectedWeightSum += factorWeight;
        }

        localDiagnostics.numTotal = finalCorrespondences.size();
        const double correspondenceCount = std::max(1, localDiagnostics.numTotal);
        localDiagnostics.meanRawResidual = absResidualSum / correspondenceCount;
        localDiagnostics.rmseRawResidual = std::sqrt(residualSquareSum / correspondenceCount);
        localDiagnostics.meanGeometryReliability = geometryReliabilitySum / correspondenceCount;
        localDiagnostics.meanResidualReliability = residualReliabilitySum / correspondenceCount;
        localDiagnostics.meanCombinedReliability = combinedReliabilitySum / correspondenceCount;
        localDiagnostics.effectiveCorrespondenceRatio = selectedWeightSum / correspondenceCount;

        lastRawInformation = 0.5 * (lastRawInformation + lastRawInformation.transpose());
        lastWeightedInformation = 0.5 * (lastWeightedInformation + lastWeightedInformation.transpose());
        localDiagnostics.rawInformationEigenvalues = sortedSelfAdjointEigenvalues(lastRawInformation);
        localDiagnostics.weightedInformationEigenvalues = sortedSelfAdjointEigenvalues(lastWeightedInformation);
        localDiagnostics.rawConditionNumber = conditionNumberFromEigenvalues(localDiagnostics.rawInformationEigenvalues);
        localDiagnostics.weightedConditionNumber = conditionNumberFromEigenvalues(localDiagnostics.weightedInformationEigenvalues);

        if (!matrixFinite(lastWeightedInformation))
        {
            localDiagnostics.usedFallback = factorCovarianceFallbackToFixed;
            if (factorCovarianceFallbackToFixed)
                setFixedLidarFactorCovarianceFallback();
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        Eigen::Matrix<double, 6, 6> information =
            lastWeightedInformation / std::max(factorNominalResidualSigma * factorNominalResidualSigma, 1e-12);
        information = 0.5 * (information + information.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> informationSolver(information);
        if (informationSolver.info() != Eigen::Success)
        {
            localDiagnostics.usedFallback = factorCovarianceFallbackToFixed;
            if (factorCovarianceFallbackToFixed)
                setFixedLidarFactorCovarianceFallback();
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        Eigen::Matrix<double, 6, 1> informationEigenvalues = informationSolver.eigenvalues();
        const double infoMin = std::max(factorInformationEigenvalueMin, 1e-12);
        const double infoMax = std::max(factorInformationEigenvalueMax, infoMin);
        const double damping = std::max(factorInformationDamping, 0.0);
        for (int i = 0; i < 6; ++i)
            informationEigenvalues(i) = std::clamp(informationEigenvalues(i), infoMin, infoMax) + damping;

        Eigen::Matrix<double, 6, 1> covarianceEigenvalues;
        for (int i = 0; i < 6; ++i)
            covarianceEigenvalues(i) = 1.0 / informationEigenvalues(i);

        const double covMin = std::max(factorCovarianceEigenvalueMin, 1e-12);
        const double covMax = std::max(factorCovarianceEigenvalueMax, covMin);
        for (int i = 0; i < 6; ++i)
            covarianceEigenvalues(i) = std::clamp(covarianceEigenvalues(i), covMin, covMax);

        Eigen::Matrix<double, 6, 6> covariance =
            informationSolver.eigenvectors() * covarianceEigenvalues.asDiagonal() * informationSolver.eigenvectors().transpose();
        covariance = 0.5 * (covariance + covariance.transpose());

        if (!factorCovarianceUseFullMatrix)
            covariance = covariance.diagonal().asDiagonal();

        if (factorCovarianceScaleMode == 1)
            covariance = blockTraceNormalizeToFixedCovariance(covariance);
        else if (factorCovarianceScaleMode == 3)
            covariance *= std::max(factorCovarianceGlobalMultiplier, 1e-12);

        const double adaptiveBlend = std::clamp(factorCovarianceAdaptiveBlend, 0.0, 1.0);
        if (adaptiveBlend < 1.0)
        {
            covariance = adaptiveBlend * covariance + (1.0 - adaptiveBlend) * fixedOdometryCovarianceLoam();
            covariance = 0.5 * (covariance + covariance.transpose());
        }

        if (!matrixFinite(covariance))
        {
            localDiagnostics.usedFallback = factorCovarianceFallbackToFixed;
            if (factorCovarianceFallbackToFixed)
                setFixedLidarFactorCovarianceFallback();
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> covarianceSolver(covariance);
        if (covarianceSolver.info() != Eigen::Success || covarianceSolver.eigenvalues().minCoeff() <= 0.0)
        {
            localDiagnostics.usedFallback = factorCovarianceFallbackToFixed;
            if (factorCovarianceFallbackToFixed)
                setFixedLidarFactorCovarianceFallback();
            lastFactorInformationDiagnostics = localDiagnostics;
            if (diagnostics)
                *diagnostics = localDiagnostics;
            return false;
        }

        lastLidarFactorCovarianceLoam = covariance;
        lastLidarFactorCovarianceGtsam = gtsam::Matrix6::Identity();
        lastLidarFactorCovarianceValid = true;
        lastLidarFactorCovarianceTimestamp = timeLaserInfoCur;
        lastLidarFactorSourceKeyframe = cloudKeyPoses3D->size();
        localDiagnostics.covarianceValid = true;
        localDiagnostics.covarianceDiagonal = covariance.diagonal();
        localDiagnostics.covarianceTrace = covariance.trace();
        localDiagnostics.covarianceRotationTrace = covariance.block<3, 3>(0, 0).trace();
        localDiagnostics.covarianceTranslationTrace = covariance.block<3, 3>(3, 3).trace();

        lastFactorInformationDiagnostics = localDiagnostics;
        if (diagnostics)
            *diagnostics = localDiagnostics;
        return true;
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAForDegeneracy(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtDegeneracy(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtADegeneracy(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            const ResidualReliability reliability =
                i < (int)reliabilitySel.size() ? reliabilitySel[i] : ResidualReliability();
            const float registrationWeight = computeRegistrationWeight(reliability.rawResidual, reliability.geometry);
            const float totalWeight = std::max(registrationWeight, 1e-3f);
            const float sqrtWeight = std::sqrt(totalWeight);

            writeReliabilityDiagnostics(
                iterCount, i, reliability, reliability.rawResidual, registrationWeight, reliability.weight, totalWeight);

            // lidar -> camera
            matAForDegeneracy.at<float>(i, 0) = arz;
            matAForDegeneracy.at<float>(i, 1) = arx;
            matAForDegeneracy.at<float>(i, 2) = ary;
            matAForDegeneracy.at<float>(i, 3) = coeff.z;
            matAForDegeneracy.at<float>(i, 4) = coeff.x;
            matAForDegeneracy.at<float>(i, 5) = coeff.y;
            matA.at<float>(i, 0) = sqrtWeight * arz;
            matA.at<float>(i, 1) = sqrtWeight * arx;
            matA.at<float>(i, 2) = sqrtWeight * ary;
            matA.at<float>(i, 3) = sqrtWeight * coeff.z;
            matA.at<float>(i, 4) = sqrtWeight * coeff.x;
            matA.at<float>(i, 5) = sqrtWeight * coeff.y;
            matB.at<float>(i, 0) = -sqrtWeight * coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            if (degeneracyHessianMode == 0)
            {
                cv::transpose(matAForDegeneracy, matAtDegeneracy);
                matAtADegeneracy = matAtDegeneracy * matAForDegeneracy;
                cv::eigen(matAtADegeneracy, matE, matV);
            }
            else
            {
                cv::eigen(matAtA, matE, matV);
            }
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            const auto registrationStart = std::chrono::steady_clock::now();
            int lmIterations = 0;
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();
                reliabilitySel.clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                lmIterations = iterCount + 1;
                if (LMOptimization(iterCount) == true)
                    break;              
            }
            const auto registrationEnd = std::chrono::steady_clock::now();

            std::array<float, 6> transformBeforeUpdate{};
            for (int i = 0; i < 6; ++i)
                transformBeforeUpdate[i] = transformTobeMapped[i];

            transformUpdate();

            const double deltaRoll = normalizedAngleDifference(transformTobeMapped[0] - transformBeforeUpdate[0]);
            const double deltaPitch = normalizedAngleDifference(transformTobeMapped[1] - transformBeforeUpdate[1]);
            const double deltaYaw = normalizedAngleDifference(transformTobeMapped[2] - transformBeforeUpdate[2]);
            const double transformUpdateDeltaRotationDeg = pcl::rad2deg(
                std::sqrt(deltaRoll * deltaRoll + deltaPitch * deltaPitch + deltaYaw * deltaYaw));
            const double dx = transformTobeMapped[3] - transformBeforeUpdate[3];
            const double dy = transformTobeMapped[4] - transformBeforeUpdate[4];
            const double dz = transformTobeMapped[5] - transformBeforeUpdate[5];
            const double transformUpdateDeltaTranslationM = std::sqrt(dx * dx + dy * dy + dz * dz);

            const auto informationStart = std::chrono::steady_clock::now();
            finalCorrespondencesValid = computeFinalCorrespondencesAtConvergedPose();
            if (!finalCorrespondencesValid)
            {
                RCLCPP_DEBUG(
                    get_logger(),
                    "Final correspondence pass found %zu correspondences, below minimum %d.",
                    finalCorrespondences.size(),
                    factorCovarianceMinCorrespondences);
            }
            computeLidarFactorInformationAndCovariance(&lastFactorInformationDiagnostics);
            const auto informationEnd = std::chrono::steady_clock::now();
            lastFactorInformationDiagnostics.lmIterations = lmIterations;
            lastFactorInformationDiagnostics.isDegenerate = isDegenerate;
            lastFactorInformationDiagnostics.registrationRuntimeMs =
                std::chrono::duration<double, std::milli>(registrationEnd - registrationStart).count();
            lastFactorInformationDiagnostics.finalInformationRuntimeMs =
                std::chrono::duration<double, std::milli>(informationEnd - informationStart).count();
            lastFactorInformationDiagnostics.transformUpdateDeltaRotationDeg = transformUpdateDeltaRotationDeg;
            lastFactorInformationDiagnostics.transformUpdateDeltaTranslationM = transformUpdateDeltaTranslationM;
            writeScanDiagnostics(cloudKeyPoses3D->size(), lastFactorInformationDiagnostics);
        } else {
            finalCorrespondences.clear();
            finalCorrespondencesValid = false;
            invalidateLidarFactorCovariance();
            FactorInformationDiagnostics diagnostics;
            diagnostics.timestamp = timeLaserInfoCur;
            diagnostics.registrationWeightMode = registrationWeightMode;
            diagnostics.factorCovarianceMode = factorCovarianceMode;
            diagnostics.factorCovarianceScaleMode = factorCovarianceScaleMode;
            diagnostics.factorCovarianceAdaptiveBlend = factorCovarianceAdaptiveBlend;
            diagnostics.numCorner = laserCloudCornerLastDSNum;
            diagnostics.numSurface = laserCloudSurfLastDSNum;
            diagnostics.isDegenerate = isDegenerate;
            lastFactorInformationDiagnostics = diagnostics;
            writeScanDiagnostics(cloudKeyPoses3D->size(), lastFactorInformationDiagnostics);
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imu_available == true && imuType)
        {
            if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    void writeReliabilityDiagnostics(
        int iterCount,
        int index,
        const ResidualReliability& reliability,
        float residual,
        float robustWeight,
        float reliabilityWeight,
        float totalWeight)
    {
        if (!reliabilityDiagnosticsFileOpen)
            return;

        const int stride = std::max(reliabilityDiagnosticsStride, 1);
        if (index % stride != 0)
            return;

        reliabilityDiagnosticsFile << std::fixed << std::setprecision(9)
                                   << timeLaserInfoCur << ","
                                   << iterCount << ","
                                   << index << ","
                                   << reliability.featureType << ","
                                   << residual << ","
                                   << robustWeight << ","
                                   << reliability.geometry << ","
                                   << reliability.residual << ","
                                   << reliabilityWeight << ","
                                   << totalWeight
                                   << '\n';
    }

    float computeHuberWeight(float residual) const
    {
        const float abs_r = std::fabs(residual);
        if (abs_r <= huberDelta)
            return 1.0f;
        return huberDelta / std::max(abs_r, 1e-6f);
    }

    float computeCauchyWeight(float residual) const
    {
        const float c = std::max(cauchyC, 1e-6f);
        const float r_over_c = residual / c;
        return 1.0f / (1.0f + r_over_c * r_over_c);
    }

    float computeRobustWeight(float residual) const
    {
        switch (robustKernelType)
        {
            case 1:
                return computeHuberWeight(residual);
            case 2:
                return computeCauchyWeight(residual);
            default:
                return 1.0f;
        }
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            const int targetKeyframeIndex = cloudKeyPoses3D->size();
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            const bool covarianceMatchesCurrentScan =
                std::abs(lastLidarFactorCovarianceTimestamp - timeLaserInfoCur) < 1e-6 &&
                lastLidarFactorSourceKeyframe == targetKeyframeIndex;
            bool useAdaptiveCovariance =
                factorCovarianceMode != 0 &&
                lastLidarFactorCovarianceValid &&
                covarianceMatchesCurrentScan;

            gtsam::SharedNoiseModel odometryNoise;
            if (useAdaptiveCovariance)
            {
                std::string covarianceError;
                const std::array<float, 6> currentLmPose = currentLmPoseArray();
                useAdaptiveCovariance = convertLmPoseCovarianceToRelativeFactorCovariance(
                    poseFrom,
                    currentLmPose,
                    lastLidarFactorCovarianceLoam,
                    &lastLidarFactorCovarianceGtsam,
                    &covarianceError);
                if (useAdaptiveCovariance)
                {
                    odometryNoise = noiseModel::Gaussian::Covariance(lastLidarFactorCovarianceGtsam);
                }
                else
                {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(),
                        *get_clock(),
                        5000,
                        "Adaptive covariance mapping failed; using fixed covariance. Reason: %s",
                        covarianceError.c_str());
                }
            }

            if (!useAdaptiveCovariance)
            {
                odometryNoise = noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            }

            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            writeKeyframeFactorDiagnostics(targetKeyframeIndex, lastFactorInformationDiagnostics, !useAdaptiveCovariance);
        }

        invalidateLidarFactorCovariance();
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (stamp2Sec(gpsQueue.front().header.stamp) < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (stamp2Sec(gpsQueue.front().header.stamp) > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
        {
            invalidateLidarFactorCovariance();
            return;
        }

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf2::Quaternion q;
        q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
        writeTrajectoryLogs(pose_in, q);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::msg::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        tf2::Quaternion quat_tf;
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        geometry_msgs::msg::Quaternion quat_msg;
        tf2::convert(quat_tf, quat_msg);
        laserOdometryROS.pose.pose.orientation = quat_msg;
        pubLaserOdometryGlobal->publish(laserOdometryROS);

        // Publish TF
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
        tf2::Stamped<tf2::Transform> temp_odom_to_lidar(t_odom_to_lidar, time_point, odometryFrame);
        geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
        tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
        trans_odom_to_lidar.child_frame_id = "lidar_link";
        br->sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imu_available == true && imuType)
            {
                if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf2::Quaternion imuQuaternion;
                    tf2::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(roll, pitch, yaw);
            geometry_msgs::msg::Quaternion quat_msg;
            tf2::convert(quat_tf, quat_msg);
            laserOdomIncremental.pose.pose.orientation = quat_msg;
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental->publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath->get_subscription_count() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath->publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, MO);

    exec.spin();

    rclcpp::shutdown();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}

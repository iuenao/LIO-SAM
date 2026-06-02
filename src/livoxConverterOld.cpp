// Converts livox_ros_driver/msg/CustomMsg bags to sensor_msgs/msg/PointCloud2.
// The old driver package is not required: the serialized wire layout matches
// livox_ros_driver2/msg/CustomMsg, so we subscribe generically and deserialize.

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp/serialization.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <string>
#include <vector>

class LivoxConverterOld : public rclcpp::Node
{
public:
    LivoxConverterOld() : Node("lio_sam_livox_converter_old")
    {
        std::vector<double> identity{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
        std::vector<double> zero{0.0, 0.0, 0.0};

        declare_parameter("pointCloudExtrinsicRot", identity);
        declare_parameter("pointCloudExtrinsicTrans", zero);
        declare_parameter("convertedPointCloudFrame", "");

        get_parameter("pointCloudExtrinsicRot", rot_);
        get_parameter("pointCloudExtrinsicTrans", trans_);
        get_parameter("convertedPointCloudFrame", converted_frame_);

        if (rot_.size() != 9)
        {
            RCLCPP_WARN(get_logger(), "pointCloudExtrinsicRot must have 9 values; using identity");
            rot_ = identity;
        }
        if (trans_.size() != 3)
        {
            RCLCPP_WARN(get_logger(), "pointCloudExtrinsicTrans must have 3 values; using zero");
            trans_ = zero;
        }

        sub_ = create_generic_subscription(
            "livox/lidar",
            "livox_ros_driver/msg/CustomMsg",
            rclcpp::SensorDataQoS(),
            std::bind(&LivoxConverterOld::callback, this, std::placeholders::_1));

        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "livox/lidar_pc2",
            rclcpp::QoS(rclcpp::KeepLast(5)).reliable());

        RCLCPP_INFO(get_logger(), "LivoxConverterOld ready: old CustomMsg -> PointCloud2");
    }

private:
    void callback(std::shared_ptr<rclcpp::SerializedMessage> raw_msg)
    {
        auto msg = std::make_shared<livox_ros_driver2::msg::CustomMsg>();
        rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg> serializer;
        serializer.deserialize_message(raw_msg.get(), msg.get());

        if (msg->point_num == 0)
            return;

        sensor_msgs::msg::PointCloud2 out;
        out.header = msg->header;
        if (!converted_frame_.empty())
            out.header.frame_id = converted_frame_;
        out.height = 1;
        out.width = msg->point_num;
        out.is_dense = false;

        sensor_msgs::PointCloud2Modifier mod(out);
        mod.setPointCloud2Fields(6,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
            "ring", 1, sensor_msgs::msg::PointField::UINT16,
            "time", 1, sensor_msgs::msg::PointField::FLOAT32);
        mod.resize(msg->point_num);

        sensor_msgs::PointCloud2Iterator<float> iter_x(out, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(out, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(out, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_i(out, "intensity");
        sensor_msgs::PointCloud2Iterator<uint16_t> iter_r(out, "ring");
        sensor_msgs::PointCloud2Iterator<float> iter_t(out, "time");

        for (const auto & pt : msg->points)
        {
            const double x = rot_[0] * pt.x + rot_[1] * pt.y + rot_[2] * pt.z + trans_[0];
            const double y = rot_[3] * pt.x + rot_[4] * pt.y + rot_[5] * pt.z + trans_[1];
            const double z = rot_[6] * pt.x + rot_[7] * pt.y + rot_[8] * pt.z + trans_[2];
            *iter_x = static_cast<float>(x);
            *iter_y = static_cast<float>(y);
            *iter_z = static_cast<float>(z);
            *iter_i = static_cast<float>(pt.reflectivity);
            *iter_r = static_cast<uint16_t>(pt.line);
            *iter_t = static_cast<float>(pt.offset_time) * 1e-9f;
            ++iter_x;
            ++iter_y;
            ++iter_z;
            ++iter_i;
            ++iter_r;
            ++iter_t;
        }

        pub_->publish(out);
    }

    rclcpp::GenericSubscription::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    std::vector<double> rot_;
    std::vector<double> trans_;
    std::string converted_frame_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxConverterOld>());
    rclcpp::shutdown();
    return 0;
}

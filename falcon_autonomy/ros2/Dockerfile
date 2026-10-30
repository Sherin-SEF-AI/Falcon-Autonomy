# Falcon's ROS 2 bridge image: the official ROS 2 Humble image (JetPack 6's ROS) with the control message packages
# Falcon's bridge node accepts. Build:  docker build -t falcon-ros2-bridge:humble falcon_autonomy/ros2
#
#   ros-humble-autoware-control-msgs   autoware_control_msgs/msg/Control (Autoware's /control/command/control_cmd)
#   ros-humble-autoware-vehicle-msgs   autoware_vehicle_msgs reports (/vehicle/status/*)
#   ros-humble-ackermann-msgs          ackermann_msgs/msg/AckermannDriveStamped
#   carla_msgs (built here)            only CarlaEgoVehicleControl.msg, copied field for field from
#                                      carla-simulator/ros-carla-msgs (MIT): steer / throttle / brake, the message
#                                      CARLA's own ROS 2 interface and its ROS bridge take
#   ros-humble-udp-msgs                udp_msgs/msg/UdpPacket (flynneva/udp_msgs, MIT): a roadside unit's CPM as UDP
#                                      payload (docs/v2x.md)
#   ros-humble-etsi-its-conversion     ika-rwth-aachen/etsi_its_messages (MIT): its conversion node decodes the UPER
#                                      CPMs into etsi_its_cpm_ts_msgs/CollectivePerceptionMessage (asn1c-generated
#                                      code); falcon v2x converter prints the command
#
# Falcon's bridge node itself is not copied in: it is mounted (or copied) from Falcon's source, see
# docs/stack-testing.md.
FROM ros:humble-ros-base

RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-humble-autoware-control-msgs ros-humble-autoware-vehicle-msgs \
        ros-humble-ackermann-msgs ros-humble-udp-msgs ros-humble-etsi-its-conversion ros-humble-etsi-its-cpm-ts-msgs \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/falcon_ws/src/carla_msgs/msg && cd /opt/falcon_ws/src/carla_msgs \
 && printf '%s\n' 'std_msgs/Header header' 'float32 throttle' 'float32 steer' 'float32 brake' 'bool hand_brake' \
        'bool reverse' 'int32 gear' 'bool manual_gear_shift' > msg/CarlaEgoVehicleControl.msg \
 && printf '%s\n' 'cmake_minimum_required(VERSION 3.8)' 'project(carla_msgs)' 'find_package(ament_cmake REQUIRED)' \
        'find_package(rosidl_default_generators REQUIRED)' 'find_package(std_msgs REQUIRED)' \
        'rosidl_generate_interfaces(${PROJECT_NAME} "msg/CarlaEgoVehicleControl.msg" DEPENDENCIES std_msgs)' \
        'ament_export_dependencies(rosidl_default_runtime)' 'ament_package()' > CMakeLists.txt \
 && printf '%s\n' '<?xml version="1.0"?>' '<package format="3">' '  <name>carla_msgs</name>' \
        '  <version>1.3.0</version>' '  <description>CarlaEgoVehicleControl only (Falcon bridge image)</description>' \
        '  <maintainer email="none@example.com">falcon</maintainer>' '  <license>MIT</license>' \
        '  <buildtool_depend>ament_cmake</buildtool_depend>' \
        '  <buildtool_depend>rosidl_default_generators</buildtool_depend>' '  <depend>std_msgs</depend>' \
        '  <exec_depend>rosidl_default_runtime</exec_depend>' \
        '  <member_of_group>rosidl_interface_packages</member_of_group>' \
        '  <export><build_type>ament_cmake</build_type></export>' '</package>' > package.xml \
 && cd /opt/falcon_ws && . /opt/ros/humble/setup.sh && colcon build --event-handlers console_direct- \
 && rm -rf build log

COPY fastdds-udp.xml /opt/falcon/fastdds-udp.xml
RUN sed -i 's|source "/opt/ros/$ROS_DISTRO/setup.bash" --|source "/opt/ros/$ROS_DISTRO/setup.bash" --\nsource /opt/falcon_ws/install/setup.bash --|' /ros_entrypoint.sh

#include <chrono>
#include <rclcpp/rclcpp.hpp>

/* ------------------  main  ------------------ */
int main(int argc,char** argv)
{
  rclcpp::init(argc,argv);
  auto node = rclcpp::Node::make_shared("block_spawner");

  rclcpp::shutdown();
  return 0;
}

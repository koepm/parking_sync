#include "parking_sync/parking_sync.hpp"
#define LEFT_CAM 2
#define RIGHT_CAM 4

ParkingSync::ParkingSync()
: Node("img_sync")
{
  img1_sub_ = std::make_shared<StampedImageMsgSubscriber>(this, "video1", rmw_qos_profile_sensor_data);
  img2_sub_ = std::make_shared<StampedImageMsgSubscriber>(this, "video2", rmw_qos_profile_sensor_data);
  lidar_flag_ = this->create_subscription<std_msgs::msg::Bool>(
    "Lidar_Stop", 10, [this](const std_msgs::msg::Bool::SharedPtr msg) {lidar_callback(msg);});

	mission_flag_ = this->create_subscription<std_msgs::msg::Int16>(
    "mission_flag", 10, [this](const std_msgs::msg::Int16::SharedPtr msg) {mission_callback(msg);});

  center_xz_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("vision/parking_points",10);

  approximate_sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<StampedImageMsg, StampedImageMsg>>>(
    message_filters::sync_policies::ApproximateTime<StampedImageMsg, StampedImageMsg>(10), *img1_sub_, *img2_sub_);

  approximate_sync_->registerCallback(std::bind(&ParkingSync::approximateSyncCallback, this, std::placeholders::_1, std::placeholders::_2));
}


void ParkingSync::lidar_callback(std_msgs::msg::Bool::SharedPtr msg)
{
	lidar_stop = msg->data;
}

void ParkingSync::mission_callback(std_msgs::msg::Int16::SharedPtr msg)
{
	mission_flag = msg->data;
	//RCLCPP_INFO(this->get_logger(), "sub message: '%s'", std::to_string(msg->data).c_str());
}

void ParkingSync::approximateSyncCallback(
  const std::shared_ptr<const sensor_msgs::msg::Image>& msg1,
  const std::shared_ptr<const sensor_msgs::msg::Image>& msg2)
{
    cv_bridge::CvImagePtr cv_ptr_left;
    cv_bridge::CvImagePtr cv_ptr_right;

    cv_ptr_left = cv_bridge::toCvCopy(msg1, "bgr8");
    cv_ptr_right = cv_bridge::toCvCopy(msg2, "bgr8");
    cv::Mat left_frame = cv_ptr_left->image;
    cv::Mat right_frame = cv_ptr_right->image;
    cv::Mat leftMask;
    cv::Mat rightMask;
    cv::Mat left_frame_copy = left_frame.clone();
    cv::Mat right_frame_copy = right_frame.clone();
    cv::Point leftCircle, rightCircle; // using yellow ball
    cv::Point2d top_XZ, mid_XZ, bottom_XZ; // using parking

    leftMask = ParkingSync::add_hsv_filter(left_frame_copy, LEFT_CAM);
    rightMask = ParkingSync::add_hsv_filter(right_frame_copy, RIGHT_CAM);
    cv::Point2f ball_XZ;
    leftCircle = ParkingSync::find_ball(left_frame, leftMask, 1);
    rightCircle = ParkingSync::find_ball(right_frame, rightMask, 2);
    ball_XZ = ParkingSync::find_xz(leftCircle, rightCircle, left_frame, right_frame, ParkingSync::alpha, ParkingSync::beta);

    //cv::resize(left_frame, left_frame, cv::Size(1280, 720));
    //cv::resize(right_frame, right_frame, cv::Size(1280, 720));

    if((mission_flag == 10)||(mission_flag == 6))
    {

      //imshow("rightMask", rightMask);
      //cv::waitKey(1);
      //imshow("leftMask", leftMask);
      //cv::waitKey(1);

      // =================[ using stereo setting ]=========================
      //Parking::line_symmetry(left_frame, LEFT_CAM);
      //Parking::line_symmetry(right_frame, RIGHT_CAM);
      //cv::rectangle(left_frame, cv::Rect(0, 0, 1280, 100), cv::Scalar(0, 0, 0), -1); //상단 for koreatech
      //cv::rectangle(right_frame, cv::Rect(0, 0, 1280, 100), cv::Scalar(0, 0, 0), -1); //상단 for koreatech
      // rectangle(left_frame, Rect(0, 400, 1280, 720), Scalar(0, 0, 0), -1); //for k-citys
      // rectangle(right_frame, Rect(0, 0, 1280, 200), Scalar(0, 0, 0), -1); //for k-citys

      //imshow("right_frame", right_frame);
      //cv::waitKey(1);
      //imshow("left_frame", left_frame);
      //cv::waitKey(1);

      // ===================================================================
      
      // =================[ using parking ]=========================
      // system("clear");
      // cout << "---------------------------------------------" << endl;
      //std::cout << "lidar_stop : " << lidar_stop << "  finish_park : " << finish_park << std::endl;	

      // imshow("right_frame", right_frame);
      // cv::waitKey(1);
      // imshow("left_frame", left_frame);
      // cv::waitKey(1);
      // =============================================================

      // =================[ using yellow ball ]=======================
      //leftMask = ParkingSync::add_hsv_filter(left_frame, LEFT_CAM);
      //rightMask = ParkingSync::add_hsv_filter(right_frame, RIGHT_CAM);
      //cv::Point2f ball_XZ;
      //leftCircle = ParkingSync::find_ball(leftMask, leftMask);
      //rightCircle = ParkingSync::find_ball(rightMask, rightMask);
      //ball_XZ = ParkingSync::find_xz(leftCircle, rightCircle, left_frame, right_frame, ParkingSync::alpha, ParkingSync::beta);
      // =============================================================
      
      // ==================[ using mouse_callback ]===================
      // img_color = left_frame.clone();
      // img_color_2 = right_frame.clone();
      // setMouseCallback("Left Frame", mouse_callback);
      // setMouseCallback("Right Frame", mouse_callback_2);

      // setMouseCallback("Left Frame",on_mouse);
      // setMouseCallback("Right Frame",on_mouse);
      // =============================================================

      if((lidar_stop == false) && (finish_park == false))
      {
        // cout << " Wating Lidar Stop" << endl;
        // imshow("Left Frame", left_frame);
        // imshow("Right Frame", right_frame);
      }
      else if((lidar_stop == true) && (finish_park == false))
      {	
        if(impulse == false)
        {
          // erp정지시 흔들림에 의해 생기는 오차 방지
          std::cout << "impulse !!" << std::endl;
          //ros::Duration(1.5).sleep(); // (구)1.5초 정지 코드
          std::chrono::milliseconds duration(1500);
          std::this_thread::sleep_for(duration);
          impulse = true;
        }

        leftMask = find_edge(left_frame, LEFT_CAM);
        rightMask = find_edge(right_frame, RIGHT_CAM);

        imshow("Left_mask", leftMask);
        imshow("Right_mask", rightMask);
        cv::waitKey(1);


        std_msgs::msg::Float64MultiArray center_xz_msg;
        ParkingSync::find_center(leftMask, left_frame, left_array, LEFT_CAM);
        ParkingSync::find_center(rightMask, right_frame, right_array, RIGHT_CAM);
        //ParkingSync::find_center(leftMask, left_frame, left_array, LEFT_CAM);
        //ParkingSync::find_center(rightMask, right_frame, right_array, RIGHT_CAM);
        
        // for(int i =0 ; i < 6; i++)
        // {
        // 	printf("배열 값[%d] : %lf\n",i, left_array[i]);
        // 	center_xz_msg.data.push_back(left_array[i]);
        // }

        // center_xz_pub_->publish(center_xz_msg);
        
        if (left_array[0] && right_array[0])
        {
          bottom_XZ = ParkingSync::find_xz({left_array[0],left_array[1]}, {right_array[0],right_array[1]}, left_frame, right_frame, ParkingSync::alpha, ParkingSync::beta);
          // mid_XZ = stereovision.`({left_array[2],left_array[3]}, {right_array[2],right_array[3]}, left_frame, right_frame, Parking::alpha, Parking::beta);
          top_XZ = ParkingSync::find_xz({left_array[4],left_array[5]}, {right_array[4],right_array[5]}, left_frame, right_frame, ParkingSync::alpha, ParkingSync::beta);
          mid_XZ.x = (bottom_XZ.x + top_XZ.x)/2.0;
          mid_XZ.y = (bottom_XZ.y + top_XZ.y)/2.0;

          std::cout << "bottom_XZ : " << bottom_XZ << std::endl;
          std::cout << "mid_XZ : " << mid_XZ << std::endl;
          std::cout << "top_XZ : " << top_XZ << std::endl;

          array[0]= (bottom_XZ.x + ParkingSync::gps_for_camera_z)/100.00;
          array[1]= -(bottom_XZ.y + ParkingSync::gps_for_camera_x)/100.00;
          array[2]= (mid_XZ.x + ParkingSync::gps_for_camera_z)/100.00;
          array[3]= -(mid_XZ.y + ParkingSync::gps_for_camera_x)/100.00;
          array[4]= (top_XZ.x + ParkingSync::gps_for_camera_z)/100.00;
          array[5]= -(top_XZ.y + ParkingSync::gps_for_camera_x)/100.00;

          //############################################ 보험용 #################################################################################################
          if((abs(array[0]-default_array[0]) > 1.3) || (abs(array[1]-default_array[1]) > 1.3) || \
          (abs(array[2]-default_array[2]) > 1.3) || (abs(array[3]-default_array[3]) > 1.3) || \
          (abs(array[4]-default_array[4]) > 1.3) || (abs(array[5]-default_array[5]) > 1.3))
          {
            std::cout << "고정점 ++" << std::endl;
            boom_count++;

            for(int i=0; i < 6; i++)
            {
              array[i] = default_array[i];
            }
          }

          for(int i=0; i<6; i++)
          {
            sum_array[i] = sum_array[i] + array[i];
          }				

          array_count++;
          std::cout << "array_count : " << array_count << std::endl;

          if(array_count == 20)
          {
            if(boom_count > 15)
            {
              std::cout << "주차실패 ㅠㅠㅠㅠ" << std::endl;
            }
            else
            {
              std::cout << " 봄 ? \n 이게 바로 비전 클라스 우리 잘못 아니니 뭐라 하려면 제어탓. \n ^~^" << std::endl;
            }
            
            std::cout << "!!!!!!!!!!!!!!!!!!" << std::endl;
            std_msgs::msg::Float64MultiArray center_xz_msg; 
            center_xz_msg.data.clear();

            for(int i=0; i<6; i++)
            {
              pub_array[i] = sum_array[i]/(double)array_count;
              center_xz_msg.data.push_back(pub_array[i]);
              printf("pub_array[%d] : %f\n", i, pub_array[i]);
            }
            
            center_xz_pub_->publish(center_xz_msg);
            finish_park = true;
            std::cout << " Finish !!!! " << std::endl;
          }
        }
      }
      else if((lidar_stop == true) && (finish_park == true))
      {
        //std::cout << " Finish !!!! " << std::endl;
      }
    }
    else
    {
      //imshow("right_frame(waiting misiion)", right_frame);
      //cv::waitKey(1);
      //imshow("left_frame(waiting misiion)", left_frame);
      //cv::waitKey(1);
    }
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ParkingSync>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}



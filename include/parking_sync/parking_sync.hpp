#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/sync_policies/exact_time.h"
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int16.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

using StampedImageMsg = sensor_msgs::msg::Image;
using StampedImageMsgSubscriber = message_filters::Subscriber<StampedImageMsg>;

class ParkingSync : public rclcpp::Node
{
public:
  ParkingSync();

private:
  std::shared_ptr<StampedImageMsgSubscriber> img1_sub_;
  std::shared_ptr<StampedImageMsgSubscriber> img2_sub_;

  void lidar_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void mission_callback(const std_msgs::msg::Int16::SharedPtr msg);
  cv::Mat undistort_frame(const cv::Mat& frame); // 이미지 왜곡 보정
  cv::Mat add_hsv_filter(const cv::Mat& frame, const int camera); // HSV 필터 적용
  cv::Point find_ball(const cv::Mat& frame, const cv::Mat& maskm, int index);
  void line_symmetry(const cv::Mat& frame, const int camera);
  double* find_center(const cv::Mat& frame, const cv::Mat& raw, double array[], const int camera);
  cv::Point2d find_xz(const cv::Point2d circle_left, const cv::Point2d circle_right, \
  const cv::Mat& left_frame, const cv::Mat& right_frame, const float alpha, const float beta);
  cv::Mat adapt_th(cv::Mat src);
  cv::Point back_parking(const cv::Mat& frame, const int camera);
  cv::Mat find_edge(const cv::Mat& frame, const int camera);
  bool isHorizontalPolygon(const std::vector<cv::Point>& vertices);
  bool isVerticalPolygon(const std::vector<cv::Point>& vertices);

  rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr mission_flag_; // 현재 진행 미션 플래그
	rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr lidar_flag_; // 라이다 스탑 플래그
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr center_xz_pub_;

  float baseline = 23;
  float focal_pixels = 800; // size(1280, 720) 일때 focal_pixels
  float alpha = 23.9;	//alpha = 카메라 머리 숙인 각도
  float beta = 45.5999; 	//beta = erp 헤딩으로부터 카메라 각도
  float gps_for_camera_x = 30; //cm
  float gps_for_camera_z = -50; //cm
  int array_count = 0;
  int boom_count = 0;
  int mission_flag = 0;
  bool lidar_stop = false;

  cv::Mat img_color;
  cv::Mat img_color_2;
  int H, S, V;
  cv_bridge::CvImagePtr cv_ptr_right;

  bool finish_park = false;
  bool impulse = false;

  double left_array[6] = {0,0,0,0,0,0};
  double right_array[6] = {0,0,0,0,0,0};
  double array[6] = {0,0,0,0,0,0};
  double pub_array[6] = {0,0,0,0,0,0};
  double default_array[6] = {2.4, -2.2, 3.65, -4.2, 4.9, -6.2}; // k-city에 맞춘 이상적인 주차구역 좌표
  float sum_array[6] ={0,0,0,0,0,0};
  cv::Point ptOld1;

  std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<StampedImageMsg, StampedImageMsg>>> approximate_sync_;
  
  void approximateSyncCallback(
    const std::shared_ptr<const sensor_msgs::msg::Image>& msg1,
    const std::shared_ptr<const sensor_msgs::msg::Image>& msg2
    );
};


#define CAM 2 // 여기에 왼쪽 카메라 인덱스 넣기

//--------------------------------------------------------------------------------------------------
/**
 * @brief 스테레오 카메라 만들 때 사용하는 함수, 초기 제작할 때 수평을 맞추기 위해서 사용함
 * @param frame 입력 영상
 * @param camera 카메라 번호 0 = 왼쪽, 1 = 오른쪽
 */
void ParkingSync::line_symmetry(const cv::Mat& frame, const int camera)
{
	rectangle(frame, cv::Rect(cv::Point(635,0),cv::Point(645,720)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,680),cv::Point(1280,690)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,640),cv::Point(1280,650)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,600),cv::Point(1280,610)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,560),cv::Point(1280,570)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,520),cv::Point(1280,530)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,480),cv::Point(1280,490)),cv::Scalar(0,255,0),2,4,0);
	rectangle(frame, cv::Rect(cv::Point(0,360),cv::Point(1280,370)),cv::Scalar(0,255,0),2,4,0);

	if(camera == CAM)
	{
		imshow("left_line_symmetry", frame);
	}
	else
	{
		imshow("right_line_symmetry", frame);
	}
}

//--------------------------------------------------------------------------------------------------
/**
 * @brief 왜곡 보정 함수
 * @param frame 입력 영상
 */
cv::Mat ParkingSync::undistort_frame(const cv::Mat& frame)
{
	// 카메라 내부 파라미터
	cv::Mat intrinsic_param = cv::Mat::zeros(3,3,CV_64FC1); // zeros로 테스트 중
	//cv::Mat intrinsic_param = Mat::eye(3,3,CV_64FC1); // eye를 쓴 이유가 뭐지?
	intrinsic_param=(cv::Mat1d(3,3) << 509.5140, 0, 321.9972, 0, 510.5093, 258.7457, 0., 0., 1. );

	// 카메라 왜곡 계수
	cv::Mat distortion_coefficient = cv::Mat::zeros(1,5,CV_64FC1);
	distortion_coefficient=(cv::Mat1d(1,5) << 0.0891, -0.1673, 0., 0., 0.);

	// 새로운 카메라 매개변수 생성
	// newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distortionParameters, { frame.cols, frame.rows }, 1);

	cv::Mat undistorted_frame;
	cv::undistort(frame, undistorted_frame, intrinsic_param, distortion_coefficient);

	return undistorted_frame;
}



//--------------------------------------------------------------------------------------------------
/**
 * @brief HSV값으로 영상을 이진화하는 함수
 * 
 * @param frame 입력하고자 하는 화면
 * @param camera 카메라 번호 0 = 왼쪽, 1 = 오른쪽
 * @return Mat 
 */
cv::Mat ParkingSync::add_hsv_filter(const cv::Mat& frame, const int camera) {

	cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
	cv::Mat mask;

	// logitech c930e
	std::vector<int> left_lowerYellow = { 10, 160, 100 };     // Lower limit for yellow
	std::vector<int> left_upperYellow = { 40, 255, 255 };	 // Upper limit for yellow
	std::vector<int> right_lowerYellow = { 10, 160, 100 };     // Lower limit for yellow
	std::vector<int> right_upperYellow = { 40, 255, 255 };	 // Upper limit for yellow

	if(camera == CAM)
	{
		inRange(frame, left_lowerYellow, left_upperYellow, mask);
	}
	else
	{
		inRange(frame, right_lowerYellow, right_upperYellow, mask);
	}
	
	return mask;
}



//--------------------------------------------------------------------------------------------------
/**
 * @brief 캘리용 함수
 * 
 * @param frame 입력 영상
 * @param mask 이진화된 영상 (HSV로 이진화하든 어쨌든 이진화된 영상)
 * @return Point 
 */
cv::Point ParkingSync::find_ball(const cv::Mat& frame, const cv::Mat& mask, int index)
{

	std::vector<std::vector<cv::Point> > contours;

	/*
	cv::RETR_EXTERNAL: 가장 외곽의 윤곽선만 검색.
	cv::CHAIN_APPROX_SIMPLE: 윤곽선 압축하여 저장. ex)직선 부분은 끝점만 저장, 곡선 부분은 시작점과 끝점 사이의 중간 점을 저장.
	*/
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	

	// Sort the contours to find the biggest one
	sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
		return contourArea(c1, false) < contourArea(c2, false);
	});

	if (contours.size() > 0) {

		std::vector<cv::Point> largestContour = contours[contours.size() - 1];
		cv::Point2f center;
		float radius;
		
		minEnclosingCircle(largestContour, center, radius);
		// 0623 아래 부분 수정함.
		//cv::Moments m = moments(largestContour);
		//cv::Point centerPoint(m.m10 / m.m00, m.m01 / m.m00);
		cv::Point centerPoint(center.x, center.y);
		// Only preceed if the radius is greater than a minimum threshold
		if (radius > 10) {
			// Draw the circle and centroid on the frame
			circle(frame, center, int(radius), cv::Scalar(0, 255, 255), 2);
			circle(frame, centerPoint, 5, cv::Scalar(0, 0, 255), -1);
		}

		if (index == 1){
			cv::imshow("left", frame);
			printf("left == x : %f, y : %f\n", center.x, center.y);
		}
		else{
			cv::imshow("right", frame);
			printf("right == x : %f, y : %f\n", center.x, center.y);
		}
		
		cv::waitKey(1);

		return centerPoint;
	}
	return { 0,0 };
}



//--------------------------------------------------------------------------------------------------
cv::Mat ParkingSync::find_edge(const cv::Mat& frame, const int camera) {


	// y값이 왜 영상의 4분의 1지점이지???
	int center_x = frame.cols /2;
	int center_y = frame.rows /4;

	// int center_x = frame.cols /2 - 150;
	// int center_y = frame.rows /4;

	cv::Mat gray_image;
	cvtColor(frame, gray_image, CV_RGB2GRAY);

	//imshow("gray_image", gray_image);
	//cv::waitKey(1);

	cv::Mat gau_image;
	GaussianBlur(gray_image,gau_image, cv::Size(3,3),3);
	//imshow("gau", gau_image);
	//cv::waitKey(1);
	
	cv::Mat dx, dy, dxdy;
	// 소벨 -> 엣지 검출
	// x방향의 엣지만 찾겠다.
	//원본 : Sobel(gray_image, dx, CV_32FC1, 1, 0); // 수평 방향 소벨 필터링
	Sobel(gray_image, dx, CV_32FC1, 1, 0); // 수평 방향 소벨 필터링
	//imshow("DX", dx);
	//cv::waitKey(1);
	// y방향의 엣지만 찾겠다.
	// 원본 : Sobel(gray_image, dy, CV_32FC1, 0, 1); // 수직 방향 소벨 필터링
	Sobel(gray_image, dy, CV_32FC1, 0, 1); // 수직 방향 소벨 필터링
	//imshow("DY", dy);
	//cv::waitKey(1);

	
	cv::Mat fmag, mag;
	magnitude(dx, dy, fmag);
	//imshow("edge1", fmag);
	//cv::waitKey(1);
	fmag.convertTo(mag, CV_8UC1);

	// 특정 강도 이상만 표시? 3기(80) 4기(50)
	//imshow("edge2", fmag);
	//cv::waitKey(1);

	cv::Mat edge = mag;
	//circle(edge, cv::Point(center_x,center_y), 10, cv::Scalar(255, 255, 255), 2);

	
	


	if(camera != CAM)
	{
		//center_x = center_x - 30;
	}

	//cv::Mat dst = cv::Mat::zeros( frame.size(), CV_8UC1 );

	// for (int r = 0; r < frame.rows; r++)
	// {
    //     for (int c = center_x; c < frame.cols-2; c++)
	// 	{
	// 		if((5*gray_image.at<uchar>(r, c) - gray_image.at<uchar>(r, c+1) - gray_image.at<uchar>(r, c+2) - gray_image.at<uchar>(r, c+3) - gray_image.at<uchar>(r, c+4) - gray_image.at<uchar>(r, c+5)) > 80) // 60
	// 		{
	// 			dst.at<uchar>(r, c-3) = 255;
	// 			dst.at<uchar>(r, c-2) = 255;
	// 			dst.at<uchar>(r, c-1) = 255;
	// 			dst.at<uchar>(r, c) = 255;
	// 			dst.at<uchar>(r, c+1) = 255;
	// 			dst.at<uchar>(r, c+2) = 255;
	// 			dst.at<uchar>(r, c+3) = 255;
	// 		}
    //     }
    // }

	// for (int r = 0; r < frame.rows; r++)
	// {
    //     for (int c = 0; c < center_x; c++)
	// 	{
	// 		if((gray_image.at<uchar>(r, c+1) + gray_image.at<uchar>(r, c+2) + gray_image.at<uchar>(r, c+3) + gray_image.at<uchar>(r, c+4) + gray_image.at<uchar>(r, c+5)) - 5*gray_image.at<uchar>(r, c) > 80) // 60
	// 		{
	// 			dst.at<uchar>(r, c-1) = 255;
	// 			dst.at<uchar>(r, c) = 255;
	// 			dst.at<uchar>(r, c+1) = 255;
	// 		}
    //     }
    // }

	// imshow("dst", dst);

	// subtract(edge, dst, edge);
	// imshow("Minus", edge);

	//circle(edge, cv::Point(center_x,center_y), 10, cv::Scalar(255, 255, 255), 2);


	dilate(edge, edge, cv::Mat(), cv::Point(-1, -1), 1);
	// erode(edge, edge, Mat(), Point(-1, -1), 1);

	//imshow("final ", edge);
	//cv::waitKey(1);

	return edge;
}


//--------------------------------------------------------------------------------------------------
/**
 * @brief 가로로 긴 사각형인지 확인
 * @param vertices 꼭짓점 좌표
 * @return bool 
 */
bool ParkingSync::isHorizontalPolygon(const std::vector<cv::Point>& points) {
    int maxX = points[0].x;
    int minX = points[0].x;

    for (const auto& point : points) {
        if (point.x > maxX)
            maxX = point.x;
        if (point.x < minX)
            minX = point.x;
    }

    int maxY = points[0].y;
    int minY = points[0].y;

    for (const auto& point : points) {
        if (point.y > maxY)
            maxY = point.y;
        if (point.y < minY)
            minY = point.y;
    }

    return (maxX - minX) > (maxY - minY);
}

//--------------------------------------------------------------------------------------------------
/**
 * @brief 세로로 긴 사각형인지 확인
 * @param vertices 꼭짓점 좌표
 * @return bool 
 */

bool ParkingSync::isVerticalPolygon(const std::vector<cv::Point>& points) {
    int maxX = points[0].x;
    int minX = points[0].x;

    for (const auto& point : points) {
        if (point.x > maxX)
            maxX = point.x;
        if (point.x < minX)
            minX = point.x;
    }

    int maxY = points[0].y;
    int minY = points[0].y;

    for (const auto& point : points) {
        if (point.y > maxY)
            maxY = point.y;
        if (point.y < minY)
            minY = point.y;
    }

    return (maxY - minY) > (maxX - minX);
}

//--------------------------------------------------------------------------------------------------
/**
 * @brief 이진화한 영상에서 중심점을 찾고 하단, 중단, 상단의 x,y좌표를 double 형의 자료형으로 저장함
 * 
 * @param img 입력 영상 (이진화된 영상을 넣으면 됨)
 * @param array double자료형의 배열(바닥 x,y, 가운데 x,y, 상단 x,y 를 저장함)
 * @param camera 카메라 번호 0 = 왼쪽, 1 = 오른쪽
 * @return double* 
 */
double* ParkingSync::find_center(const cv::Mat& frame, const cv::Mat& raw, double array[], const int camera)
{
	int center_x = frame.cols / 2;
	int center_y = frame.rows / 4;

	//cv::Mat mask;
	//cv::cvtColor(frame, mask, cv::COLOR_BGR2GRAY);

	// Ensure the image is of type CV_8UC1
	cv::threshold(frame, frame, 127, 255, cv::THRESH_BINARY);

	// 외곽 윤곽선에 중요한 점 저장
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(frame, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat contourImage = cv::Mat::zeros(frame.size(), CV_8UC3);
	cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 0, 255), 2);
	
	// Show the contour image
	if(camera == CAM){
		cv::imshow("Contours1", contourImage);
		cv::waitKey(1);
	}
	

	// // 모든 윤곽선에 중요한 점 저장(계층o)
	// std::vector<std::vector<cv::Point>> contours2;
	// std::vector<cv::Vec4i> hierarchy2;
	// cv::findContours(frame, contours2, hierarchy2, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	// cv::Mat contourImage2 = cv::Mat::zeros(frame.size(), CV_8UC3);
	// cv::drawContours(contourImage2, contours2, -1, cv::Scalar(0, 0, 255), 2);

	// // 모든 윤곽선에 중요한 점 저장(계층x)
	// std::vector<std::vector<cv::Point>> contours3;
	// std::vector<cv::Vec4i> hierarchy3;
	// cv::findContours(frame, contours3, hierarchy3, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	// cv::Mat contourImage3 = cv::Mat::zeros(frame.size(), CV_8UC3);
	// cv::drawContours(contourImage3, contours3, -1, cv::Scalar(0, 0, 255), 2);

	// // 모든 점 저장
	// std::vector<std::vector<cv::Point>> contours4;
	// std::vector<cv::Vec4i> hierarchy4;
	// cv::findContours(frame, contours4, hierarchy4, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	// cv::Mat contourImage4 = cv::Mat::zeros(frame.size(), CV_8UC3);
	// cv::drawContours(contourImage4, contours4, -1, cv::Scalar(0, 0, 255), 2);
	

	
	// cv::imshow("Contours2", contourImage2);
	// cv::waitKey(1);
	// cv::imshow("Contours3", contourImage3);
	// cv::waitKey(1);
	// cv::imshow("Contours4", contourImage4);
	// cv::waitKey(1);

    double nowarea = 0;
	double min_area = 5000;
	cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
    int lrgctridx = 0;
	int min_distance = 1000;

	//std::vector<cv::Point2f> approx;

	for (size_t i = 0; i < contours.size(); i++)
	{
		std::vector<cv::Point> approx;
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);

		// approx.size() >= 3 && approx.size() <= 5 &&
		if (cv::isContourConvex(approx)) // 교선이 없는 닫힌 다각형만 처리
		{
			// Calculate the area of the polygon
			double area = cv::contourArea(approx);

			//if (area > min_area && approx.size() == 4)
			if (area > min_area && approx.size() == 4) 
			{

				size_t topLeftIndex = 0;
				int topLeftSum = approx[0].x + approx[0].y;

				for (size_t j = 1; j < approx.size(); j++) 
				{
					int currentSum = approx[j].x + approx[j].y;
					if (currentSum < topLeftSum) {
						topLeftIndex = j;
						topLeftSum = currentSum;
					}
				}

				// Reorder the vertices to start from the top-left vertex
				std::rotate(approx.begin(), approx.begin() + topLeftIndex, approx.end());
				
				// Draw the polygon
				std::vector<std::vector<cv::Point>> contour;
				contour.push_back(approx);

				cv::drawContours(raw, contour, -1, cv::Scalar(0, 0, 255), 2);


				for (size_t j = 0; j < approx.size(); j++) 
				{
                	cv::circle(raw, approx[j], 5, cv::Scalar(0, 255, 0), -1);

					// Calculate the index of the next vertex
					size_t nextIndex = (j + 1) % approx.size();

					// Calculate the midpoint between the current and next vertex
					cv::Point midpoint = (approx[j] + approx[nextIndex]) / 2;

					// Draw the midpoint on the image
					cv::circle(raw, midpoint, 3, cv::Scalar(255, 0, 0), -1);

					// Draw the index number
					std::stringstream ss;
					ss << j;
					cv::putText(raw, ss.str(), midpoint, cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 255), 2);
            	}

				cv::Moments m = cv::moments(cv::Mat(approx), true);
				cv::Point center(m.m10 / m.m00, m.m01 / m.m00);

				// Draw the center point on the image
				cv::circle(raw, center, 5, cv::Scalar(0, 0, 255), -1);

				// if (isVerticalPolygon(approx)){
				// 	printf("세로\n");
				// }
				// else if (isHorizontalPolygon(approx)){
				// 	printf("가로\n");
				// }
				// else {
				// 	printf("nothing\n");
				// }
				// Display the image
				if (camera == CAM){
					cv::imshow("Polygon_left1", raw);
					cv::waitKey(1);
				}
				

				
			}

			// 꼭짓점이 3개일 경우 프레임 모서리에 나머지 꼭짓점 생성해서 사용
			// else if (area > min_area && approx.size() == 3)
			// {
			// 	return 0;
			// }

			/* 5개이상의 꼭짓점이 있을 경우 화면 모서리에 가장 가까운 점 4개만 사용 
			else if (area > min_area && approx.size() >= 5)
			{
				// Find the four closest points to the top left, bottom left, top right, and bottom right corners
				cv::Point topLeft = approx[0];
				cv::Point bottomLeft = approx[0];
				cv::Point topRight = approx[0];
				cv::Point bottomRight = approx[0];

				for (size_t j = 1; j < approx.size(); j++)
				{
					if (approx[j].x + approx[j].y < topLeft.x + topLeft.y)
						topLeft = approx[j];

					if (approx[j].x + approx[j].y > bottomRight.x + bottomRight.y)
						bottomRight = approx[j];

					if (approx[j].x - approx[j].y < topRight.x - topRight.y)
						topRight = approx[j];

					if (approx[j].x - approx[j].y > bottomLeft.x - bottomLeft.y)
						bottomLeft = approx[j];
				}

				// Check if any of the four points lie inside the convex hull of the other three points
				cv::Point center = (topLeft + bottomLeft + topRight + bottomRight) / 4;
				bool isConvex = cv::pointPolygonTest(approx, center, false) > 0;

				// Create a rectangle only if the points do not form a convex shape
				if (!isConvex)
				{
					std::vector<cv::Point> rect;
					rect.push_back(topLeft);
					rect.push_back(bottomLeft);
					rect.push_back(topRight);
					rect.push_back(bottomRight);

					// Draw the rectangle with different colors for the calculated and existing vertices
					std::vector<std::vector<cv::Point>> contour;
					contour.push_back(rect);

					cv::drawContours(raw, contour, -1, cv::Scalar(0, 0, 255), 2);
					for (size_t j = 0; j < approx.size(); j++)
					{
						if (approx[j] != topLeft && approx[j] != bottomLeft && approx[j] != topRight && approx[j] != bottomRight)
							cv::circle(raw, approx[j], 5, cv::Scalar(0, 0, 255), -1);
					}

					// Display the image
					cv::imshow("Polygon", raw);
					cv::waitKey(1);
				}
			}
			*/
				
		}
		
		//cv::Moments m = moments(contours[i], true);
		//cv::Point p(m.m10/m.m00, m.m01/m.m00);

		cv::Moments m = cv::moments(cv::Mat(approx), true);
		cv::Point p(m.m10 / m.m00, m.m01 / m.m00);
		
		std::vector<cv::Point> hull;
        cv::convexHull(cv::Mat(contours[i]), hull, false);

		nowarea = cv::contourArea(cv::Mat(hull));

		// cout << nowarea << endl;

		//if(approx.size() > 3)
		if(approx.size() >= 4 && isContourConvex(approx))
		{	
			//if ((nowarea > 35000) && (nowarea < 100000))	//35000	// 일단 25000이었음
			if ((nowarea > 5000) && (nowarea < 100000))	//35000	// 일단 25000이었음
			{
				if(sqrt(pow((p.x-center_x),2)+pow((p.y-center_y),2)) < min_distance)
				{
					min_distance = sqrt(pow((p.x-center_x),2)+pow((p.y-center_y),2));
					lrgctridx = i;
					// cout << " 넓이 : " << nowarea << endl;
				}


				cv::Mat drawing = cv::Mat::zeros( frame.size(), CV_8UC3 );

				// 모든 외각선 그리기
				// for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
				// {
				//     drawContours( drawing, contours, idx, color, 2, LINE_8, hierarchy);
				// }

				// 특정한 외각선만 그리기
				drawContours( drawing, contours, lrgctridx, color, 2, cv::LINE_8, hierarchy);

				std::vector<cv::Point> hull;
				cv::convexHull(cv::Mat(contours[lrgctridx]), hull, false);

				std::vector<std::vector<cv::Point>> fake_hull;
				fake_hull.push_back(hull);
				drawContours(drawing, fake_hull, 0, color, 2, cv::LINE_8);

				int top_x_left = hull[0].x;
				int top_y_left = hull[0].y;
				int top_num_left = 0;

				int bottom_x_left = hull[0].x;
				int bottom_y_left = hull[0].y;
				int bottom_num_left = 0;

				int top_x_right = hull[0].x;
				int top_y_right = hull[0].y;
				int top_num_right = 0;

				int bottom_x_right = hull[0].x;
				int bottom_y_right = hull[0].y;
				int bottom_num_right = 0;

				for(int i = 0; i < int(hull.size()); i++)
				{
					// if(hull[i].y < top_y_left)
					// {
					//     top_x_left = hull[i].x;
					//     top_y_left = hull[i].y;
					//     top_num_left = i;
					// }
					if(sqrt(pow(hull[i].x - frame.cols*7/8, 2) + pow(hull[i].y - 0, 2)) < sqrt(pow(top_x_right - frame.cols*7/8, 2) + pow(top_y_right - 0, 2)))
					{
						top_x_right = hull[i].x;
						top_y_right = hull[i].y;
						top_num_right = i;
					}
					if((1*sqrt(pow(hull[i].x - 0, 2) + pow(hull[i].y - frame.rows, 2)) + 9*hull[i].x) < (1*sqrt(pow(bottom_x_left - 0, 2) + pow(bottom_y_left - frame.rows, 2)) + 9*bottom_x_left))
					{
						bottom_x_left = hull[i].x;
						bottom_y_left = hull[i].y;
						bottom_num_left = i;
					}
					if(sqrt(pow(hull[i].x - frame.cols, 2) + pow(hull[i].y - frame.rows, 2)) < sqrt(pow(bottom_x_right - frame.cols, 2) + pow(bottom_y_right - frame.rows, 2)))
					{
						bottom_x_right = hull[i].x;
						bottom_y_right = hull[i].y;
						bottom_num_right = i;
					}
				}

				double daegaksun = sqrt(pow(bottom_x_left - top_x_right, 2) + pow(bottom_y_left - top_y_right, 2));
				double long_sin = 0;
				double fake_sin;

				for(int j=0; j < int(hull.size()); j++)
				{
					if((hull[j].y < bottom_y_left -50) && (hull[j].x < top_x_right - 50) && (hull[j].x < bottom_x_right - 50))
					{
						double sasun1 = sqrt(pow(hull[j].x - bottom_x_left, 2) + pow(hull[j].y - bottom_y_left, 2));
						double sasun2 = sqrt(pow(hull[j].x - top_x_right, 2) + pow(hull[j].y - top_y_right, 2));

						double theta = acos((pow(sasun1, 2) - pow(sasun2, 2) + pow(daegaksun, 2)) / (2*sasun1*daegaksun));

						fake_sin = sasun1 * sin(theta);

						if(fake_sin > long_sin)
						{
							long_sin = fake_sin;

							top_x_left = hull[j].x;
							top_y_left = hull[j].y;
						}

					}
				}

				double mean_top_x = (double)((top_x_left + top_x_right) / 2.0);
				double mean_top_y = (double)((top_y_left + top_y_right) / 2.0);
				double mean_mid_x = (double)((top_x_left+top_x_right+bottom_x_left+bottom_x_right)/4.0);
				double mean_mid_y = (double)((top_y_left+top_y_right+bottom_y_left+bottom_y_right)/4.0);
				double mean_bottom_x = (double)((bottom_x_left + bottom_x_right) / 2.0);
				double mean_bottom_y = (double)((bottom_y_left + bottom_y_right) / 2.0);

				circle(drawing, cv::Point((int)mean_top_x, (int)mean_top_y), 10, cv::Scalar(0, 0, 255), -1);
				circle(drawing, cv::Point((int)mean_mid_x, (int)mean_mid_y), 10, cv::Scalar(0, 255, 0), -1);
				circle(drawing, cv::Point((int)mean_bottom_x, (int)mean_bottom_y), 10, cv::Scalar(255, 0, 0), -1);
				
				circle(drawing, cv::Point(top_x_left, top_y_left), 4, cv::Scalar(0, 0, 255), -1);
				//printf("x : %d y : %d\n",top_x_left, top_y_left);
				circle(drawing, cv::Point(top_x_right, top_y_right), 4, cv::Scalar(0, 255, 0), -1);
				circle(drawing, cv::Point(bottom_x_left, bottom_y_left), 4, cv::Scalar(255, 0, 0), -1);
				circle(drawing, cv::Point(bottom_x_right, bottom_y_right), 4, cv::Scalar(255, 255, 255), -1);

				array[0] = mean_bottom_x;
				array[1] = mean_bottom_y;
				array[2] = mean_mid_x;
				array[3] = mean_mid_y;
				array[4] = mean_top_x;
				array[5] = mean_top_y;

				if(camera == CAM)
				{
					imshow("Polygon_left2", drawing);
					cv::waitKey(1);
				}
				
				return array;
			}
		}
	}
}



//--------------------------------------------------------------------------------------------------
/**
 * @brief 실제 erp의 GPS를 기준으로 X, Z값을 계산하는 함수
 * 
 * left_point와 right_point는 서로 대응되는 점을 넣어야 함
 * ex) {left_array[0],left_array[1]}를 넣으면 {right_array[0],right_array[1]}를 넣어야 함
 * 
 * @param left_point find_center 함수로 구한 좌측 카메라에서의 x,y값
 * @param right_point find_center 함수로 구한 우측 카메라에서의 x,y값
 * @param left_frame 왼쪽화면 입력영상
 * @param right_frame 오른쪽화면 입력영상
 * @param alpha 카메라가 고개를 숙인 각도 (처음 find_ball함수를 이용해 캘리를 하면서 구함)
 * @param beta 카메라가 틀어진 각도 (처음 find_ball함수를 이용해 캘리를 하면서 구함)
 * @return Point2d (실제 X거리값, 실제 Z거리값, cm단위임)
 */
cv::Point2d ParkingSync::find_xz(const cv::Point2d circle_left, const cv::Point2d circle_right, \
const cv::Mat& left_frame, const cv::Mat& right_frame, const float alpha, const float beta)
{
	float x_0 = 0;
	float y_0 = 0;

	if ((right_frame.cols == left_frame.cols) && (right_frame.rows == left_frame.rows))
	{	
		x_0 = right_frame.cols/2;
		y_0 = right_frame.rows/2;
	}
	else {
		std::cout << "Left and Right Camera frames do not have the same pixel width" << std::endl;	
	}

	float xLeft = circle_left.x;
	float xRight = circle_right.x;
	float yLeft = circle_left.y;
	float yRight = circle_right.y;

	float realX = 0;
	float realY = 0;
	float realZ = 0;
	float distance = 0;

	if(xLeft != x_0)
	{
		realX = (float)ParkingSync::baseline/(1 - (x_0 - xRight)/(x_0 - xLeft));
		realZ = abs(realX*ParkingSync::focal_pixels/(x_0 - xLeft));
	}
	else if(xRight != x_0)
	{
		realX = -(float)ParkingSync::baseline/(1 - (x_0 - xLeft)/(x_0 - xRight));
		realZ = abs(realX*ParkingSync::focal_pixels/(x_0 - xRight));
		realX = realX + (float)ParkingSync::baseline; //왼쪽 카메라 기준
	}
	else
	{
		realX = 0;
		realY = 0;
	}
	realY = realZ*(2*y_0-yLeft-yRight)/(2*ParkingSync::focal_pixels);

	distance = sqrt(pow(realX,2)+pow(realY,2) + pow(realZ,2));

	//std::cout << " realX : " << realX << "   realY : "<< realY << "     realZ : " << realZ << std::endl << std::endl;
	// cout << " distance : " << distance << endl;
	
	// cout << " 영점 조절 : " << realX << endl;
	
	//ERP 기준 좌표로 변환
	ParkingSync::alpha = ParkingSync::alpha * CV_PI / 180;
	ParkingSync::beta = ParkingSync::beta * CV_PI / 180;

	float fakeZ = realZ;
	float fakeY = realY;

	float theta = atan(fakeY/fakeZ) - ParkingSync::alpha;
	realZ = sqrt(pow(fakeZ,2)+pow(fakeY,2))*cos(theta);
	realY = sqrt(pow(fakeZ,2)+pow(fakeY,2))*sin(theta);

	float realZ_copy = realZ;
	float realX_copy = realX;

	float gama = atan(realX/realZ) + ParkingSync::beta;
	realZ = sqrt(pow(realZ_copy,2)+pow(realX_copy,2))*cos(gama);
	realX = sqrt(pow(realZ_copy,2)+pow(realX_copy,2))*sin(gama);
	
	// float angle = 0;
	// angle = atan(realX/realZ)*180/CV_PI;

	std::cout << "realZ : " << realZ << "  realX : " << realX << std::endl;
	return {realZ, realX};
}



//--------------------------------------------------------------------------------------------------
/**
 * @brief 그림자 문제를 해결하기 위해 적응형 이진화 함수를 사용해 색공간이 아닌 다른 방법으로 영상을 이진화함
 * 
 * @param src 입력영상 
 * @return Mat 이진화된 영상 
 */
cv::Mat ParkingSync::adapt_th(cv::Mat src)
{
	cv::Mat image;
	cv::Mat binary;

	image = src.clone();

	resize(image, image, cv::Size(640, 480));

	imshow("cap", image);

	cvtColor(image, binary, CV_BGR2GRAY);
	// namedWindow("dst");
	// createTrackbar("Block_Size", "dst", 0, 200, on_trackbar, (void*)&binary);
	// setTrackbarPos("Block_Size", "dst", 11);

	adaptiveThreshold(binary, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 9, 5);

	cv::Mat binary_inv;

	binary_inv = ~binary;

	// morphologyEx(binary, binary, MORPH_OPEN, Mat(), Point(-1, -1), 3);

	morphologyEx(binary_inv, binary_inv, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);

	erode(binary_inv, binary_inv, cv::Mat());

	resize(binary_inv, binary_inv, cv::Size(640, 480));

	imshow("dst", binary_inv);

	return binary_inv;
}

//-------------------------  사용 안하는 코드!!!  -------------------------------------
//void line_symmetry(const cv::Mat& frame, const int camera);
/**
 * @brief 평행주차용으로 개발했으나 라바콘이 예상과는 다르게 사용되어 쓸 일이 없음
 * 
 * @param img 
 * @param camera 
 * @return Point
 */
/*
Point StereoVision::back_park(Mat &img, int camera)
{
	Mat mask = img.clone();

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	dilate(mask, mask, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 2);

	findContours(mask, contours, hierarchy,RETR_LIST, CHAIN_APPROX_SIMPLE);

	Scalar color(rand() & 255, rand() & 255, rand() & 255);

    int lrgctridx = 0;
    int minarea = 1000000;
	int maxarea = 0;
    double nowarea = 0;
	bool find_contour = false;
	Point contour_moment = {0,0};
	int minDistance = 10000;

	for (size_t i = 0; i < contours.size(); i++)
	{
		nowarea = contourArea(contours[i]);
		Moments m = moments(contours[i], true);
		Point p(m.m10/m.m00, m.m01/m.m00);

		float d = sqrt(pow(p.x-img.cols*5/6,2)+pow(p.y-img.rows*5/6,2));

		// if ((nowarea > 4000) && (nowarea < 30000) && d > 300)
		// {
		// 	if(nowarea > maxarea)
		// 	{
		// 		maxarea = nowarea;
		// 		lrgctridx = i;
		// 		contour_moment = p;
		// 		find_contour = true;
		// 		// cout << " size : " << nowarea << endl;
		// 	}
		// }
		// cout << "!!! : " << sqrt(pow(img.cols/6,2)+pow(img.rows/6,2)) << endl;
		if ((nowarea > 2000) && (nowarea < 30000) && (d > 250))
		{
			if(d < minDistance)
			{
				minDistance = d;
				lrgctridx = i;
				contour_moment = p;
				find_contour = true;
				// cout << " size : " << nowarea << endl;
				// cout << " d : " << d << endl;
			}
		}
	}

	int bottom_x = 0;
	int bottom_y = 0;

	if(find_contour == true)
	{
		Mat drawing = Mat::zeros( mask.size(), CV_8UC3 );

		// drawContours( mask, contours, lrgctridx, Scalar(255, 255, 255), 2, LINE_8, hierarchy);

        vector<Point> hull;
        convexHull(Mat(contours[lrgctridx]), hull, false);

        vector<vector<Point>> fake_hull;
        fake_hull.push_back(hull);
        drawContours(drawing, fake_hull, 0, color, 4, LINE_8);

		bottom_x = hull[0].x;
        bottom_y = hull[0].y;

		for(int i = 0; i < hull.size(); i++)
        {
            if(hull[i].y > bottom_y)
            {
                bottom_x = hull[i].x;
                bottom_y = hull[i].y;
            }
        }

		circle(drawing, {bottom_x,bottom_y}, 8, Scalar(0, 255, 0), -1);

        circle(drawing, contour_moment, 8, Scalar(0, 0, 255), -1);

		if(camera == 0)
		{
			imshow("Left", drawing);
		}
		else
		{
			imshow("Right", drawing);
		}
		
		return {bottom_x,bottom_y}; // contour_moment
    }
    else
    {
        return {bottom_x,bottom_y}; //contour_moment
    }
}
*/

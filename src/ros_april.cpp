#include <iostream>
#include <ctype.h>
#include <sstream>
#include <math.h>
#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;



#include <condition_variable>
#include <mutex>
#include <thread>


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>

#include <cv_bridge/cv_bridge.h>

#include "ros/ros.h"

#include <tf/tf.h>
#include <wave_utils/wave_math.h>

// April tags detector and various families that can be selected by command line option
#include "AprilTags/TagDetector.h"
#include "AprilTags/Tag16h5.h"
#include <april_tag_one_target_ips/ips_msg.h>

#include <tf/transform_broadcaster.h>

//#define DEBUG_KF_TEST
#define DEBUG_ROS_APRIL
//#define DEBUG_APRIL_PROFILING
//#define DEBUG_APRIL_LATENCY_PROFILING

ros::Publisher pose_pub;
tf::TransformBroadcaster *br;
tf::Transform *tform;


//
// SYDE FYDP Group: Make changes to the parameters here.
//
#define IMAGE_SCALING 0.5f // Set to 1 for no image resizing

double m_tagSize(0.154);

#define TAG_ID 5

// Camera calibration
double m_fx(691.828717 * IMAGE_SCALING);
double m_fy(691.649664 * IMAGE_SCALING);
double m_px(345.823544 * IMAGE_SCALING);
double m_py(269.292294 * IMAGE_SCALING);
cv::Vec4f __distParam(-0.392347, 0.240234, -0.005363, 0.001404);

// To set the origin:
// 1. Run the program with the origin_position set to (0,0,0)
// and origin_orientation set to (0,0,0,1).
// 2. Place the tag at the desired origin and get the position and 
// orientation of the tag.
// 3. Put those measured values into origin_position and origin_orientation,
// as is.
// 4. The origin offset will be done automatically.
Eigen::Vector3d __origin_position(-0.206602007151, 0.211576685309, -2.95268201828);
tf::Quaternion origin_orientation(0.044971, -0.21773,-0.650177,0.726527);

tf::Matrix3x3 __origin_rotation_matrix(origin_orientation);

//
// End of parameters that need modifications.
//


#define REL_WINDOW_SIZE 2.0f // Window size in multiples of m_tagSize

// Enter


AprilTags::TagCodes m_tagCodes = AprilTags::tagCodes16h5;

AprilTags::TagDetector* m_tagDetector;


geometry_msgs::Point window_centre_point; // Last tag pose in camera frame
bool crop_image = false;
ros::Time prev_t;

//camera orientation to body orientation
Eigen::Matrix3d cToM;


/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
  yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
  double c = cos(yaw);
  double s = sin(yaw);
  pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
  roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
}


void print_detection(AprilTags::TagDetection& detection) {
  
  if(detection.id != TAG_ID)
    return;

  // recovering the relative pose of a tag:
  // NOTE: for this to be accurate, it is necessary to use the
  // actual camera parameters here as well as the actual tag size
  // (m_fx, m_fy, m_px, m_py, m_tagSize)

  Eigen::Vector3d tag_translation; //in camera frame
  Eigen::Matrix3d tag_rotation;    //in camera frame
  detection.getRelativeTranslationRotation(m_tagSize, m_fx, m_fy, m_px, m_py,
                                           tag_translation, tag_rotation);

  double tag_roll, tag_pitch, tag_yaw;
  wRo_to_euler(tag_rotation, tag_yaw, tag_pitch, tag_roll);

  // Note: Roll measures 180 degrees at a neutral attitude for some reason.
  // I know this isn't yaw, but the yawWrap function does the bounding we want.
  tag_roll = wave::yawWrap(tag_roll + M_PI);

  Eigen::Vector3d qr_frame_tag_angles(tag_roll, tag_pitch, tag_yaw);

  Eigen::Matrix3d rotation_roll_pi = wave::createEulerRot(M_PI, 0, 0);
  qr_frame_tag_angles = rotation_roll_pi * qr_frame_tag_angles;

  Eigen::Matrix3d rotation_yaw_half_pi = wave::createEulerRot(0, 0, M_PI/2);
  qr_frame_tag_angles = rotation_yaw_half_pi * qr_frame_tag_angles;

  tf::Quaternion tag_quaternion;
  tag_quaternion.setEulerZYX(qr_frame_tag_angles(2), qr_frame_tag_angles(1),
                        qr_frame_tag_angles(0));

  tf::Matrix3x3 tag_dcm(tag_quaternion);
  Eigen::Vector3d copterTranslation = cToM*tag_translation;
  double measured_x_position = -copterTranslation(2);
  double measured_y_position = -copterTranslation(1);
  double measured_z_position = -copterTranslation(0);


  Eigen::Matrix4d measured_pose;
  measured_pose <<
    tag_dcm[0][0], tag_dcm[0][1], tag_dcm[0][2], measured_x_position,
    tag_dcm[1][0], tag_dcm[1][1], tag_dcm[1][2], measured_y_position,
    tag_dcm[2][0], tag_dcm[2][1], tag_dcm[2][2], measured_z_position,
    0, 0, 0, 1;
    

  // Correct for origin offset.
  Eigen::Matrix4d origin_pose;
  origin_pose <<
    __origin_rotation_matrix[0][0], __origin_rotation_matrix[0][1], __origin_rotation_matrix[0][2], __origin_position(0),
    __origin_rotation_matrix[1][0], __origin_rotation_matrix[1][1], __origin_rotation_matrix[1][2], __origin_position(1),
    __origin_rotation_matrix[2][0], __origin_rotation_matrix[2][1], __origin_rotation_matrix[2][2], __origin_position(2),
    0,                              0,                              0,                              1;

  Eigen::Matrix4d measured_pose_with_origin_correction;
  measured_pose_with_origin_correction = origin_pose.inverse() * measured_pose;

  tf::Matrix3x3 rotation_with_origin_correction(measured_pose_with_origin_correction(0,0),
    measured_pose_with_origin_correction(0,1), measured_pose_with_origin_correction(0,2),
    measured_pose_with_origin_correction(1,0), measured_pose_with_origin_correction(1,1),
    measured_pose_with_origin_correction(1,2), measured_pose_with_origin_correction(2,0),
    measured_pose_with_origin_correction(2,1), measured_pose_with_origin_correction(2,2));

  tf::Quaternion final_quaternion;

  rotation_with_origin_correction.getRotation(final_quaternion);

  geometry_msgs::Pose to_publish;
  to_publish.position.x = measured_pose_with_origin_correction(0, 3);
  to_publish.position.y = measured_pose_with_origin_correction(1, 3);
  to_publish.position.z = measured_pose_with_origin_correction(2, 3);
  tf::quaternionTFToMsg(final_quaternion, to_publish.orientation);

  cout<<"x: "<<to_publish.orientation.x<<", y: "<<to_publish.orientation.y<<", z:" << to_publish.orientation.z<<", w: "<<to_publish.orientation.w<<endl;

  prev_t = ros::Time::now();
  crop_image = true;

  april_tag_one_target_ips::ips_msg curpose;
  curpose.header.stamp = ros::Time::now();
  curpose.tag_id = 0;
  curpose.hamming_distance = 0;
  curpose.header.frame_id="/map";
  curpose.X = to_publish.position.x;
  curpose.Y = to_publish.position.y;
  curpose.Z = to_publish.position.z;
  curpose.Roll = 0; //tf::getRoll(to_publish.orientation);;
  curpose.Pitch = 0;//tf::getPitch(to_publish.orientation);;
  curpose.Yaw = tf::getYaw(to_publish.orientation); // Robot Yaw
  //republish pose for rviz

  // send transform
  br = new tf::TransformBroadcaster;
  tform = new tf::Transform;
  tform->setOrigin( tf::Vector3(to_publish.position.x, to_publish.position.y, 0) );
  tf::Quaternion q;
  q.setEulerZYX(tf::getYaw(to_publish.orientation), 0,0);
  tform->setRotation( q );
  *tform = tform->inverse();
  br->sendTransform(tf::StampedTransform(*tform, ros::Time::now(), "base_footprint", "map"));

  pose_pub.publish(curpose);

/*
#ifdef DEBUG_ROS_APRIL
  double DEG2RAD = 180.0 / M_PI;
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  ROS_INFO("Euler Angles: %f, %f, %f", roll*DEG2RAD, pitch*DEG2RAD, yaw*DEG2RAD);
#endif
*/
}


// Converts a coordinate in meters in the camera frame into the image's pixel coordinates.
// The returned point is not guaranteed to be in the image.
// @return point corresponding with the specified coordinate in the pixel frame.
cv::Point reproject_camera_frame_to_image_frame(double cf_x, double cf_y, double cf_z)
{
  // Note x and y are swapped from the camera frame to image frame

  double x_prime = cf_y / cf_z;
  double y_prime = cf_x / cf_z;

  double r_squared = pow(x_prime, 2) + pow(y_prime, 2);

  double x_prime2 = x_prime*(1 + __distParam(0)*r_squared + __distParam(1)*pow(r_squared,2)) +
        2*__distParam(2)*x_prime*y_prime + __distParam(3)*(r_squared + 2*pow(x_prime,2));
  double y_prime2 = y_prime*(1 + __distParam(0)*r_squared + __distParam(1)*pow(r_squared,2)) +
        __distParam(2)*(r_squared + 2*pow(y_prime,2)) + 2*__distParam(3)*x_prime*y_prime;

  return cv::Point(m_fx*x_prime2 + m_px, m_fy*y_prime2 + m_py);
}


// Returns a vector of two points, the top left corner of the ROI, then the bottom right corner of
// the ROI. Must maintain this order.
vector<cv::Point> calculate_roi_from_pose_estimate(geometry_msgs::Point tag_pose)
{
  vector<cv::Point> rect_corners;

  // Now do the inverse calculation. Taken from OpenCV Camera Calibration page.
  rect_corners.push_back(
        reproject_camera_frame_to_image_frame(tag_pose.x + m_tagSize*REL_WINDOW_SIZE, 
        tag_pose.y + m_tagSize*REL_WINDOW_SIZE, tag_pose.z) );
  
  rect_corners.push_back(
        reproject_camera_frame_to_image_frame(tag_pose.x - m_tagSize*REL_WINDOW_SIZE,
        tag_pose.y - m_tagSize*REL_WINDOW_SIZE, tag_pose.z) );
  
  return rect_corners;
}

// This function fixes the roi boundary points in place, or signals if the ROI
// cannot be fixed because too much of it is off the screen.
// Input: The vector returned by calculate_roi_from_pose_estimate()
// Return: true if the ROI is good or corrected to be good, false otherwise
bool fix_roi_boundaries(cv::Mat& image, vector<cv::Point>& roi_points)
{
  assert(2 == roi_points.size());

  // Check that upper left corner is not beyond the right and bottom bounds
  // of the image
  if ( (roi_points[0].x > image.cols)
        || (roi_points[0].y > image.rows) )
  {
    return false;
  }

  // Check that upper right corner is not beyond the left and top bounds
  // of the image
  if ( (roi_points[1].x < 0)
        || (roi_points[1].y < 0) )
  {
    return false;
  }

  // Check initial size, we want to make sure the ROI does not shrink by more
  // than some threshold percentage.
  int dx_pre = abs(roi_points[1].x - roi_points[0].x);
  int dy_pre = abs(roi_points[1].y - roi_points[0].y);


  // Fix upper left corner of ROI to be at least the upper left corner of image
  if ( roi_points[0].x < 0 )
  {
    roi_points[0].x = 0;
  }
  if ( roi_points[0].y < 0 )
  {
    roi_points[0].y = 0;
  }

  // Fix bottom right corner of ROI to be at least the bottom right corner of image
  if ( roi_points[1].x >= image.cols )
  {
    roi_points[1].x = image.cols - 1;
  }
  if ( roi_points[1].y >= image.rows )
  {
    roi_points[1].y = image.rows - 1;
  }

  // Check dimensions and make sure it didn't shrink too much
  int dx_post = abs(roi_points[1].x - roi_points[0].x);
  int dy_post = abs(roi_points[1].y - roi_points[0].y);
  if ( ( ( (float) dx_post / (float) dx_pre ) < 0.85f ) 
        || ( ( (float) dy_post / (float) dy_pre ) < 0.85f ) )
  {
    return false;
  }
  
  return true;
}


bool new_tag_pos_prediction = false;

void tag_pos_pred_cb(geometry_msgs::Point point)
{
//  new_tag_pos_prediction = true;
//  window_centre_point = point;
}


// TODO: This is currently a dirty, filthy, no-good duplicate of the callback.
void run_april_tag_detection_and_processing(cv::Mat& image_gray)
{
#ifdef DEBUG_APRIL_LATENCY_PROFILING
  // Need a new way to profile
#endif //DEBUG_APRIL_LATENCY_PROFILING

#ifdef DEBUG_APRIL_PROFILING
  static int n_count = 0;
  static double t_accum = 0;

  ros::Time start = ros::Time::now();

  if ( (ros::Time::now() - prev_t).toSec() > 1.0 )
  {
    crop_image = false;
  }
#endif // DEBUG_APRIL_PROFILING

#ifdef DEBUG_ROS_APRIL
  bool curr_frame_cropped = false;
#endif

  vector<AprilTags::TagDetection> detections;
  vector<cv::Point> rect_corners;

  if (!new_tag_pos_prediction)
  {
    crop_image = false;
  }
  // Clear flag so we don't window the same place again
  new_tag_pos_prediction = false;

  if (crop_image)
  {
    rect_corners = calculate_roi_from_pose_estimate(window_centre_point);
    crop_image = fix_roi_boundaries(image_gray, rect_corners);
  }

  detections= m_tagDetector->extractTags(image_gray);


  for(int i = 0; i < (int) detections.size(); i++)
  {
    print_detection(detections[i]);
  }

#ifdef DEBUG_ROS_APRIL
  for (int i=0; i < (int) detections.size(); i++) {
    // also highlight in the image
    if(detections[i].id == 0)
    {
      detections[i].draw(image_gray);
    }
  }

  if (curr_frame_cropped)
  {
    cv::rectangle(image_gray, rect_corners[0], rect_corners[1], cv::Scalar(0,255,255), 3);
  }

  imshow("AprilResult", image_gray); // OpenCV call
  cv::waitKey(30);
#endif

#ifdef DEBUG_APRIL_PROFILING
  ros::Time end = ros::Time::now();
  n_count++;
  t_accum += (end - start).toSec();
  if (n_count >= 100)
  {
    ROS_DEBUG("Avg april tag run time: %f", t_accum/100.0);
    std::cerr << "Avg april tag run time: " << t_accum/100.0 << std::endl;
    n_count = 0;
    t_accum = 0;
  }
#endif // DEBUG_APRIL_PROFILING

#ifdef DEBUG_APRIL_LATENCY_PROFILING
  // Need a new way to profile
#endif //DEBUG_APRIL_LATENCY_PROFILING

}

void image_callback(const sensor_msgs::ImageConstPtr& picture)
{
  cv_bridge::CvImageConstPtr bridge = cv_bridge::toCvShare(picture,"mono8");
  cv_bridge::CvImage thresholdPublisher;
  
  cv::Mat image_gray;
  cv::resize(bridge->image, image_gray, cv::Size(), IMAGE_SCALING,
             IMAGE_SCALING);

  run_april_tag_detection_and_processing(image_gray);
}


int main( int argc, char** argv )
{
  //ROS init
  ros::init(argc, argv, "ros_april");
  ros::NodeHandle n("~");

  bool use_default_camera = true;

  // Does not disable ueye camera if not using it's message... need to fix that.
  std::string image_topic;
  if(ros::names::remap("image") == "image")
  {
    ROS_INFO("image has not been mapped, using default /camera/image");
    image_topic = std::string("/camera/image");
  }
  else
  {
    use_default_camera = false;
    image_topic = std::string(ros::names::remap("image"));
  }


  cToM <<
    1, 0, 0,  //right of camera points to the right
    0,-1, 0, //Bottom of camera points backward
    0, 0,-1; //Front of camera faces down


  m_tagDetector = new AprilTags::TagDetector(m_tagCodes);

  //pose_pub = n.advertise<geometry_msgs::Pose>("/wave/tag", 1);
  pose_pub = n.advertise<april_tag_one_target_ips::ips_msg>("/indoor_pos", 1, true);
  ros::Subscriber image_sub = n.subscribe(image_topic, 1, image_callback,ros::TransportHints().tcpNoDelay());
 
#ifdef DEBUG_KF_TEST
  ros::Subscriber tag_kf_pred_sub = n.subscribe("/wave/tag_pos_pred", 1, tag_pos_pred_cb, ros::TransportHints().tcpNoDelay());
#else
  ros::Subscriber tag_kf_pred_sub = n.subscribe("/wave/tag_window", 1, tag_pos_pred_cb, ros::TransportHints().tcpNoDelay());
#endif

  prev_t = ros::Time::now();

  ROS_INFO("Starting tag detection.");

  ros::spin();

  return 0;
}


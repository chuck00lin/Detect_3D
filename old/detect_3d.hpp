#ifndef DETECT_3D_HPP
#define DETECT_3D_HPP

#define GLFW_INCLUDE_GLU
#define OPENCV

#include <QMainWindow>
#include <librealsense2/rs.hpp>
#include "opencv.hpp"
#include "opencv2/dnn.hpp"
#include <yolo_v2_class.hpp>
#include <GLFW/glfw3.h>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <algorithm>

#include <QString>
#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QElapsedTimer>

#include "detect_with_depth.hpp"

struct bbox_t_history:bbox_t
{
    QList<unsigned int> history;                      // Fruit state history sequence: 0: Inactive, 1: Lost, 2: Tracked
    QList<cv::Point2f> trajectory;                    // Fruit center x, y coordinate when in Tracked state
    unsigned int lost_frame;                          // Frame number when the fruit is Lost
    QList<cv::Mat> frame_mat;                         // Save row frame for maturity calculation
    QList<cv::Point2d>  width_height;                 // Fruit width and height
    QList<double> depth_hist;                         // Fruit distance when in Tracked state
    QList<std::pair<float, float>> true_size_hist;    // Fruit true size calculated by 3D point cloud (NOT USED IN THE NEWEST VERSION) (width, height (m))
    double maturity;                                  // Fruit maturity. Calculated in offline stage
    int size;                                         // Fruit size in pixel value. Calculated in offline stage
    double median_depth;                              // Fruit depth calculate by the median value of all pixels in online stage runtime. Used for saving in QList<double> depth_hist.
    std::pair<float, float> true_size;                // Fruit true size calculated by 3D point cloud in online stage runtime. Used for saving in QList<std::pair<float, float>>  true_size_hist (NOT USED IN THE NEWEST VERSION)
};

struct global_coor{
    cv::Point2f global_point;       // Fruit global coordinate calculated in offline stage
    unsigned int global_fruit_ID;   // Fruit ID, same as bbox_t_history.track_ID
    double maturity;                // Fruit maturity, same as bbox_t_history.maturity
    int size;                       // Fruit size, same as bbox_t_history.size
    int nearest_trajectory_index;   // Fruit trajectory index with smallest distance between the fruit and camera (NOT USED IN THE NEWEST VERSION)
};




namespace Ui {
class detect_3d;
}

class detect_3d : public QMainWindow
{
    Q_OBJECT

public:
    explicit detect_3d(QWidget *parent = 0);
    ~detect_3d();

private slots:
    void on_save_stateChanged(int arg1);

    void on_distance_clicked();

    void on_point_cloud_clicked();

    void on_actionexit_triggered();


    void on_camera_pose_clicked();

    void on_save_pose_stateChanged(int arg1);

    void on_scale_valueChanged(int arg1);

    void on_frame_valueChanged(int arg1);

    void on_epipolar_clicked();

    void on_fast_track_clicked();

    void on_save_iou_stateChanged(int arg1);

    void on_depth_filter_clicked();

private:
    Ui::detect_3d *ui;
    bool save_dis;
    bool save_pose;
    bool saveIOU;
    QString dis_path;
    QString pose_path;
    int scale;
    int frame;

};

#endif // DETECT_3D_HPP

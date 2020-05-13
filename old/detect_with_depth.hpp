#ifndef DETECT_WITH_DEPTH_HPP
#define DETECT_WITH_DEPTH_HPP

#include <QObject>
#include <QDebug>

#include <QMainWindow>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
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
#include <queue>
#include <mutex>
#include <math.h>
#include <time.h>

#include <QString>
#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QElapsedTimer>

#include "detect_3d.hpp"


class detect_with_depth : public QWidget
{
    Q_OBJECT
public:
    detect_with_depth();
    void get_bag_file();
    void initialize_realsense();
    void initialize_detector(bool);

    inline void Color();
    inline void Depth();
    inline void draw_color();
    inline void draw_depth();
    inline void showColor();
    inline void showDepth();

public slots:
    void run();
    void set_saveiou(int);
private:
    // Realsense
    rs2::pipeline pipeline;
    rs2::pipeline_profile pipeline_profile;
    rs2::frameset frameset;

    // Color Buffer
    rs2::frame color_frame;
    cv::Mat color_mat;
    uint32_t color_width;
    uint32_t color_height;

    // Depth Buffer
    rs2::frame depth_frame;
    cv::Mat depth_mat;
    uint32_t depth_width;
    uint32_t depth_height;

    // Video
    std::string bag_filename;

    // YOLO detector
    std::string cfg_file;
    std::string weights_file;
    std::string names_file;

    bool save_IOU;

    // Thread
};

#endif // DETECT_WITH_DEPTH_HPP

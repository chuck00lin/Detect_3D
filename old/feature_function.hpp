#ifndef FEATURE_FUNCTION_HPP
#define FEATURE_FUNCTION_HPP

#define HAVE_OPENCV_XFEATURES2D

#include "opencv2/opencv_modules.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"

#include "opencv2/core/version.hpp"
#include "opencv2/videoio/videoio.hpp"

#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

//OPENCV CUDA
#include "opencv2/core.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

#include <yolo_v2_class.hpp>
#include <detect_3d.hpp>

#include <QString>

using namespace cv;
using namespace cv::cuda;

inline void featureDetection(cv::Mat img, std::vector<cv::Point2f>& point1, int count){

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(700);
    std::vector<cv::KeyPoint> keypoint1;
    detector->detect(img, keypoint1);

    cv::Mat img_keypoints_1;
    cv::drawKeypoints(img, keypoint1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::KeyPoint::convert(keypoint1, point1, std::vector<int>());
}

inline void featureDetection_GPU(cv::Mat img, std::vector<cv::Point2f>& point1, int count){
    cv::cuda::GpuMat img1;
    cv::cvtColor(img, img, CV_BGR2GRAY);
    img1.upload(img);
    cv::cuda::SURF_CUDA surf;
    cv::cuda::GpuMat keypointGPU;
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    surf(img1, GpuMat(), keypointGPU);
//    qDebug() << "Found " << keypointGPU.cols << "keypoints";
    std::vector<cv::KeyPoint> keypoint1;
    surf.downloadKeypoints(keypointGPU, keypoint1);
    cv::KeyPoint::convert(keypoint1, point1, std::vector<int>());
}

inline float featureTracking(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f>& point1, std::vector<cv::Point2f>& point2, std::vector<uchar>& status, rs2::depth_frame depth, bool depthornot){
    std::vector<float> err;
    cv::Size winSize = cv::Size(50, 50);
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.01);
    cv::calcOpticalFlowPyrLK(img1, img2, point1, point2, status, err, winSize, 3, termcrit, 0, 0.001);
    int indexCorrection = 0;
    for(int i = 0 ; i < status.size() ; i++){
        cv::Point2f pt = point2.at(i - indexCorrection);
        if((status.at(i) == 0 ) || (pt.x < 0) || (pt.y < 0)){
            if((pt.x < 0) || (pt.y < 0)){
                status.at(i) = 0;
            }
            point1.erase(point1.begin() + i - indexCorrection);
            point2.erase(point2.begin() + i - indexCorrection);
            indexCorrection++;
        }
    }
    if(depthornot){
        float dist_sum = 0;
        int img_w = depth.get_width();
        int img_h = depth.get_height();
        for(int i = 0 ; i < point1.size() ; i++){
            int x = (int)point1.at(i).x;
            int y = (int)point1.at(i).y;
            //            qDebug() << x << y;
            if(x < img_w && y < img_h)
                dist_sum += depth.get_distance(x, y);
        }
        float avg = dist_sum / point1.size();
        qDebug() << "dist_sum" << dist_sum << "avg" << avg;
        return avg;
    }
    else
        return 0.0;
}

inline float featureTracking_GPU(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f>& point1, std::vector<cv::Point2f>& point2, std::vector<uchar>& status, rs2::depth_frame depth, bool depthornot){

    cv::Size winSize = cv::Size(21, 21);
    int maxLevel = 3;
    int iters = 30;

    cv::cvtColor(img1, img1, CV_BGR2GRAY);
    cv::cvtColor(img2, img2, CV_BGR2GRAY);

    cv::cuda::GpuMat d_frame0Gray(img1);
    cv::cuda::GpuMat d_prevPts;
    cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(d_frame0Gray.type(), 10000, 0.01, 0);

    detector->detect(d_frame0Gray, d_prevPts);

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK = cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel, iters);

    cv::cuda::GpuMat d_frame0(img1);
    cv::cuda::GpuMat d_frame1(img2);
    cv::cuda::GpuMat d_nextPts;
    cv::cuda::GpuMat d_status;
    d_pyrLK->calc(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);

    point1.resize(d_prevPts.cols);
    cv::Mat mat(1, d_prevPts.cols, CV_32FC2, (void*)&point1[0]);
    d_prevPts.download(mat);

    point2.resize(d_nextPts.cols);
    cv::Mat mat1(1, d_nextPts.cols, CV_32FC2, (void*)&point2[0]);
    d_nextPts.download(mat1);

    qDebug() << "***** INFUNCTION *****" << d_prevPts.cols << d_nextPts.cols;
    qDebug() << "Before: " << point1.size();
    qDebug() << "After: " << point2.size();


    // Filter feature points: only keep feature points within depth limits (0.3m~0.9m)
    int count = 0;
    if(depthornot){
        float dist_sum = 0;
        int img_w = depth.get_width();
        int img_h = depth.get_height();
        double dist_1, dist_2;
        for(int i = 0 ; i < point1.size() ; i++){
            int x1 = (int)point1.at(i).x;
            int y1 = (int)point1.at(i).y;
            int x2 = (int)point2.at(i).x;
            int y2 = (int)point2.at(i).y;

            if((x1 < img_w && y1 < img_h) && (x1 >= 0 && y1 >= 0)){
                dist_sum += depth.get_distance(x1, y1);
                dist_1 = depth.get_distance(x1, y1);
                if((x2 < img_w && y2 < img_h) && (x2 >= 0 && y2 >= 0)){
                    dist_2 = depth.get_distance(x2, y2);
                    if((dist_1 < 0.3 || dist_1 > 0.9) || (dist_2 < 0.3 || dist_2 > 0.9)){
                        point1.erase(point1.begin()+i);
                        point2.erase(point2.begin()+i);
                        count++;
                        if(count > 9000)    break;
                        i--;
                    }
                }
            }

        }
        float avg = dist_sum / point1.size();
        qDebug() << "dist_sum" << dist_sum << "avg" << avg;
        qDebug() << "After: " << point1.size();
        qDebug() << "After: " << point2.size();
        return avg;
    }
    else
        return 0.0;
}

inline cv::Point2f get_tracked_point(cv::Mat homography_matrix, cv::Point2f input_point){

    double M11, M12, M13, M21, M22, M23;
    if(homography_matrix.rows == 0 || homography_matrix.cols == 0){
        M11 = 1; M12 = 0; M13 = 0;
        M21 = 0; M22 = 1; M23 = 0;
    }
    else{
        M11 = homography_matrix.at<double>(0, 0);
        M12 = homography_matrix.at<double>(0, 1);
        M13 = homography_matrix.at<double>(0, 2);
        M21 = homography_matrix.at<double>(1, 0);
        M22 = homography_matrix.at<double>(1, 1);
        M23 = homography_matrix.at<double>(1, 2);
    }

    //    qDebug() << "input: " << input_point.x << input_point.y;
    //    qDebug() << "         " << M11 << M12 << M13;
    //    qDebug() << "         " << M21 << M22 << M23;

    float x = (float)input_point.x;
    float y = (float)input_point.y;
    cv::Point2f temp_point(((M11 * (float)x + M12 * (float)y + M13))
                           , ((M21 * (float)x + M22 * (float)y + M23)));

    return temp_point;
}

inline cv::Point2f global_coordinate(cv::Mat homography_matrix, cv::Point2f input_point){

    double M11, M12, M13, M21, M22, M23;
    if(homography_matrix.rows == 0 || homography_matrix.cols == 0){
        M11 = 1; M12 = 0; M13 = 0;
        M21 = 0; M22 = 1; M23 = 0;
    }
    else{
        M11 = homography_matrix.at<double>(0, 0);
        M12 = homography_matrix.at<double>(0, 1);
        M13 = homography_matrix.at<double>(0, 2);
        M21 = homography_matrix.at<double>(1, 0);
        M22 = homography_matrix.at<double>(1, 1);
        M23 = homography_matrix.at<double>(1, 2);
    }

    // Inverse Matrix
    // cv::Mat inverse_homo = homography_matrix;
    // double M11 = inverse_homo.at<double>(0, 0);
    // double M12 = inverse_homo.at<double>(0, 1);
    // double M13 = inverse_homo.at<double>(0, 2);
    // double M21 = inverse_homo.at<double>(1, 0);
    // double M22 = inverse_homo.at<double>(1, 1);
    // double M23 = inverse_homo.at<double>(1, 2);

    float x = (float)input_point.x;
    float y = (float)input_point.y;

    cv::Point2f temp_point(((float)(M11 * x + (float)M12 * y + M13))
                           , (((float)M21 * x + (float)M22 * y + M23)));

    cv::Point2f vector(temp_point.x - input_point.x, temp_point.y - input_point.y);
    cv::Point2f result(input_point.x - vector.x, input_point.y - vector.y);

    return result;
    //    return temp_point;  // Inverse Matrix
}

inline void set_ID(std::vector<bbox_t_history>& total_fruit, QList<cv::Point> prev_tracked_fruit, QList<cv::Point> curr_fruit, std::vector<bbox_t_history>& prev_vec, std::vector<bbox_t_history>& curr_vec, int threshold, bool prev_fruit){
    for(int i = 0 ; i < curr_fruit.size() ; i++){
        if(prev_fruit){
            double distance = cv::norm(prev_tracked_fruit.at(0) - curr_fruit.at(i));    // Distance between predict point and true point
            if(distance < threshold){
                curr_vec.at(i).track_id = prev_vec.at(0).track_id;
            }
            for(int j = 1 ; j < prev_tracked_fruit.size() ; j++){
                double temp = cv::norm(prev_tracked_fruit.at(j) - curr_fruit.at(i));
                if(temp < distance) {
                    distance = temp;
                    if(temp < threshold) curr_vec.at(i).track_id = prev_vec.at(j).track_id;
                    else curr_vec.at(i).track_id = 0;
                }
            }
            if(curr_vec.at(i).track_id == 0){   // New Fruit compare to last frame
                curr_vec.at(i).track_id = total_fruit.size() + 1;
                total_fruit.push_back(curr_vec.at(i));
            }
        }
        else{
            curr_vec.at(i).track_id = total_fruit.size() + 1;
            total_fruit.push_back(curr_vec.at(i));
        }
    }
}

inline double IOU(bbox_t_history prev_vec, bbox_t_history curr_vec, cv::Mat homo){

    cv::Point2f prev_start((float)prev_vec.x, (float)prev_vec.y);
    cv::Point2f curr_start((float)curr_vec.x, (float)curr_vec.y);
    cv::Point2f prev_end((float)prev_vec.x + (float)prev_vec.w, (float)prev_vec.y + (float)prev_vec.h);
    cv::Point2f curr_end((float)curr_vec.x + (float)curr_vec.w, (float)curr_vec.y + (float)curr_vec.h);

    cv::Point2f start1 = get_tracked_point(homo, prev_start);
    cv::Point2f end1 = get_tracked_point(homo, prev_end);
    cv::Point2f start2 = curr_start;
    cv::Point2f end2 = curr_end;

    double width = std::min(end1.x, end2.x) - std::max(start1.x, start2.x);
    double height = std::min(end1.y, end2.y) - std::max(start1.y, start2.y);

    double intersection_area = width * height;
    double IOU;
    if(width < 0 || height < 0) IOU = 0.0;
    else IOU = intersection_area / (((end1.x - start1.x) * (end1.y - start1.y)) + (curr_vec.w * curr_vec.h) - intersection_area);

    qDebug() << "IOU: " << IOU;

    return IOU;
}

inline double IOU(cv::Point2f prev_pt_LT, cv::Point2f curr_pt_LT, cv::Point2f prev_pt_RB, cv::Point2f curr_pt_RB){

    double curr_w = curr_pt_RB.x - curr_pt_LT.x;
    double curr_h = curr_pt_RB.y - curr_pt_LT.y;
    double prev_w = prev_pt_RB.x - prev_pt_LT.x;
    double prev_h = prev_pt_RB.y - prev_pt_LT.y;

    double width = std::min(prev_pt_RB.x, curr_pt_RB.x) - std::max(prev_pt_LT.x, curr_pt_LT.x);
    double height = std::min(prev_pt_RB.y, curr_pt_RB.y) - std::max(prev_pt_LT.y, curr_pt_LT.y);

    double intersection_area = width * height;
    double IOU;
    if(width < 0 || height < 0) IOU = 0.0;
    else IOU = intersection_area / (curr_w * curr_h + prev_w * prev_h - intersection_area);
    qDebug() << "2 IOU: " << IOU;
    return IOU;
}


inline void set_ID_fast(std::vector<bbox_t_history>& total_fruit
                        , std::vector<bbox_t_history>& prev_vec, std::vector<bbox_t_history>& curr_vec
                        , QList<cv::Mat> Homo_history, QList<double> mean_depth_diff
                        , bool prev_fruit, QList<int> threshold, int lost_track_threshold, QList<float> avg_point_dist_hist
                        , cv::Mat& check_mat, cv::Mat maturity_mat
                        , bool save_IOU, bool depth)
{
    double IOU_threshold = 0.0;
    cv::Mat save_frame = maturity_mat.clone();
    int curr_frame = Homo_history.size();

    QList<int> used_ids, closed;
    int origin_threshold = 0;

    int used_id = -1;

    // ------------------ STAGE ONE POLICY ------------------ //
    for(int i = 0 ; i < curr_vec.size() ; i++){
        cv::Point2f trajectory((float)curr_vec.at(i).x + (float)curr_vec.at(i).w / 2, (float)curr_vec.at(i).y + (float)curr_vec.at(i).h / 2);
        cv::Point2d w_h(curr_vec.at(i).w, curr_vec.at(i).h);
        cv::Point2f curr_fruit((float)curr_vec[i].x + (float)curr_vec[i].w / 2, (float)curr_vec[i].y + (float)curr_vec[i].h / 2);

        double closeness_threshold = 1000;
        int closed_index = -1;
        for(int ch = 0 ; ch < curr_vec.size() ; ch++){
            if(ch != i){
                cv::Point2f too_closed((float)curr_vec.at(ch).x + (float)curr_vec.at(ch).w / 2, (float)curr_vec.at(ch).y + (float)curr_vec.at(ch).h / 2);
                double closeness = cv::norm(curr_fruit - too_closed);
                if(closeness < closeness_threshold) {
                    closeness_threshold = closeness;
                    closed_index = ch;
                }
            }
        }
        if(prev_fruit){
            double distance = 1000;    // Distance between predict point and true point
            for(int j = 0 ; j < prev_vec.size() ; j++){

                origin_threshold = std::min(threshold.at(j), 50);
                if(closeness_threshold < 50){
                    closed.append(closed_index);
                    origin_threshold /= 2;
                    cv::circle(check_mat, curr_fruit, origin_threshold, cv::Scalar(0, 255, 0), 2);
                }

                bool duplicate = false;
                for(int n = 0 ; n < used_ids.size() ; n++){
                    if(prev_vec.at(j).track_id == used_ids.at(n)){
                        duplicate = true;
                        break;
                    }
                }
                if(duplicate){
                    continue;
                }
                cv::Point2f p((float)prev_vec.at(j).x + (float)prev_vec.at(j).w / 2, (float)prev_vec.at(j).y + (float)prev_vec.at(j).h / 2);
                cv::Point2f prev_tracked_fruit = get_tracked_point(Homo_history.at(curr_frame - 1), p);

                double temp = cv::norm(prev_tracked_fruit - curr_fruit);
                double iou = IOU(prev_vec.at(j), curr_vec.at(i), Homo_history.at((curr_frame - 1)));
                double depth_diff = std::abs((double)prev_vec.at(j).median_depth - (double)curr_vec.at(i).median_depth);

                QFile history;
                QTextStream out(&history);
                if(save_IOU){
                    history.setFileName("./depth_data/info_fruit.csv");
                    if(history.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
                        out << "Frame: " << curr_frame << ", L2-norm: "
                            << temp << ", IOU: " << iou << ", " << "Depth-diff: "
                            << depth_diff << ", Current-vs-x|y: (" << curr_vec.at(i).x << "|" << curr_vec.at(i).y
                            << "), (" << prev_vec.at(j).x << "|" << prev_vec.at(j).y << ")\n";
                    }
                    history.close();
                }
                if((temp < distance) && (iou > IOU_threshold) && (depth_diff < 0.05)) {
                    distance = temp;
                    if(temp < origin_threshold){
                        curr_vec.at(i).track_id = prev_vec.at(j).track_id;
                        used_id = prev_vec.at(j).track_id;
                        qDebug() << "first stage - tracked  ID:" << prev_vec.at(j).track_id;
                    }
                    else {
                        curr_vec.at(i).track_id = 0;
                        qDebug() << "first stage - Lost2Track or New";}
                }
            }

            if(used_id != -1)    used_ids.append(used_id);
            if(curr_vec.at(i).track_id != 0){   // Old Fruit compare to last frame
                auto it = std::find_if(total_fruit.begin(), total_fruit.end(), [&](bbox_t_history &vector)
                { return vector.track_id == curr_vec.at(i).track_id; });

                it->trajectory.append(trajectory);
                it->history.append(2);
                it->frame_mat.append(save_frame);
                it->width_height.append(w_h);
                it->true_size_hist.append(curr_vec.at(i).true_size);
                it->depth_hist.append(curr_vec.at(i).median_depth);
            }
        }
        else{
            curr_vec.at(i).track_id = total_fruit.size() + 1;
            for(int pp = 0 ; pp < curr_frame ; pp++){
                curr_vec.at(i).history.append(0);    // Inactive
            }
            curr_vec.at(i).history.append(2);   // First Tracked
            curr_vec.at(i).trajectory.append(trajectory);
            curr_vec.at(i).frame_mat.append(save_frame);
            curr_vec.at(i).width_height.append(w_h);
            curr_vec.at(i).true_size_hist.append(curr_vec.at(i).true_size);
            curr_vec.at(i).depth_hist.append(curr_vec.at(i).median_depth);
            total_fruit.push_back(curr_vec.at(i));
            qDebug() << "2. New Fruit (No Fruit in last frame)  ID: " << curr_vec.at(i).track_id;
        }

    }

    // ------------------ Append history with lost and mark the lost frame ------------------ //
    for(int i = 0 ; i < total_fruit.size() ; i++){
        if(total_fruit.at(i).history.at(total_fruit.at(i).history.size() - 1) != 0){ // Not inactive
            if(total_fruit.at(i).history.size() < curr_frame + 1){
                total_fruit.at(i).history.append(1);    // Lost
                if(total_fruit.at(i).history.at(total_fruit.at(i).history.size() - 2) == 2){   // From tracked -> lost
                    total_fruit.at(i).lost_frame = curr_frame;
                }
            }
        }
    }

    // ------------------ STAGE TWO POLICY ------------------ //
    for(int i = 0 ; i < curr_vec.size() ; i++){
        if(curr_vec.at(i).track_id == 0){   // 1. Lost -> Tracked  2. New fruit

            QList<double> dis;
            QList<double> IOU_r;
            QList<int> lost_index;
            QList<double> accu_depth_diff;
            cv::Point2f curr_point((float)curr_vec.at(i).x + (float)curr_vec.at(i).w / 2, (float)curr_vec.at(i).y + (float)curr_vec.at(i).h / 2);
            cv::Point2d w_h(curr_vec.at(i).w, curr_vec.at(i).h);
            double curr_depth = curr_vec.at(i).median_depth;
            qDebug() << "check : 1. Lost -> Tracked  2. New fruit";
            for(int j = 0 ; j < total_fruit.size() ; j++){
                if(total_fruit.at(j).history.at(total_fruit.at(j).history.size() - 1) == 1){    // Lost
                    int th = sqrt(pow(total_fruit.at(j).h, 2) + pow(total_fruit.at(j).w, 2)) / 2;
                    double iou;
                    cv::Point2f lost_point = total_fruit.at(j).trajectory.at(total_fruit.at(j).trajectory.size() - 1);
                    cv::Point2f lost_wh = total_fruit.at(j).width_height.at(total_fruit.at(j).width_height.size() - 1);
                    cv::Point2f lost_left_top(lost_point.x - lost_wh.x / 2, lost_point.y - lost_wh.y / 2);
                    cv::Point2f lost_right_bottom(lost_point.x + lost_wh.x / 2, lost_point.y + lost_wh.y / 2);
                    double lost_depth = total_fruit.at(j).median_depth;
                    cv::putText(check_mat, "ID : " + std::to_string(total_fruit.at(j).track_id), cv::Point2f(lost_point.x - 30, lost_point.y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), 1.5);
                    qDebug() << "=== Search for Lost fruit, Lost ID: " << total_fruit.at(j).track_id
                             << ", lost frame: " << total_fruit.at(j).lost_frame
                             << ", lost point: " << lost_point.x << ", " << lost_point.y
                             << ", homo_size: " << Homo_history.size()
                             << ", lost depth: " << lost_depth << " ===";
                    bool track_is_out = false;
                    for(int h = total_fruit.at(j).lost_frame - 1 ; h <= curr_frame - 1 ; h++){
                        if(depth)   lost_depth += mean_depth_diff.at(h);

                        float avg_point_dist = avg_point_dist_hist.at(h);
                        float prev_fruit_dist = total_fruit.at(j).median_depth;
//                        float ratio = prev_fruit_dist / avg_point_dist; // (m/m)
                                                float ratio = 1; // (m/m)

                        cv::Point2f output = get_tracked_point(Homo_history.at(h), lost_point);
                        if(depth){
                            float x_diff = output.x - lost_point.x;
                            float y_diff = output.y - lost_point.y;
                            output.x = x_diff * ratio + lost_point.x;
                            output.y = y_diff * ratio + lost_point.y;
                        }


                        cv::Point2f output_lost_left_top = get_tracked_point(Homo_history.at(h), lost_left_top);
                        cv::Point2f output_lost_right_bottom = get_tracked_point(Homo_history.at(h), lost_right_bottom);

                        // If lost-tracked point is out of the img --> From Lost to Inactive && won't compare with others
                        if(output.x < 0 || output.x > check_mat.cols || output.y < 0 || output.y > check_mat.rows){
                            track_is_out = true;
                            total_fruit.at(j).history.pop_back();
                            total_fruit.at(j).history.append(0);
                            break;
                        }
                        lost_point = output;
                        lost_left_top = output_lost_left_top;
                        lost_right_bottom = output_lost_right_bottom;
                        cv::circle(check_mat, lost_point, 3, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), -1);
                    }
                    qDebug() << "CURRENT DEPTH: "<< curr_depth
                             << ", LOST DEPTH AFTER CALCULATE: " << lost_depth;
                    if(track_is_out == false){      // If tracked point is inside the img --> put into compare
                        int line_width = 1;
                        cv::circle(check_mat, lost_point, th, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), line_width);
                        double temp = cv::norm(lost_point - curr_point);
                        cv::Point2f curr_left_top(curr_vec.at(i).x, curr_vec.at(i).y);
                        cv::Point2f curr_right_bottom(curr_vec.at(i).x + curr_vec.at(i).w, curr_vec.at(i).y + curr_vec.at(i).h);
                        iou = IOU(lost_left_top, curr_left_top, lost_right_bottom, curr_right_bottom);
                        double diff_depth;
                        if(depth)   {diff_depth = std::abs(lost_depth - curr_depth);}
                        qDebug() << "dist: "<< temp;
                        dis.append(temp);
                        IOU_r.append(iou);
                        if(depth)   {accu_depth_diff.append(diff_depth);}
                        lost_index.append(j);  // j means lost fruit in total_fruit with "INDEX" not "ID"
                    }
                    //  cv::putText(check_mat, "IOU : " + std::to_string(iou), cv::Point2f(lost_point.x - 30, lost_point.y + 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar((j*10)%255, (j*50)%255, (j*100)%255), 1.5);
                }
            }
            if(dis.size() != 0){    // There are Lost fruits in total fruit
                double min = dis.at(0);
                int index = lost_index.at(0);
                for(int k = 1 ; k < dis.size() ; k++){
                    if(depth){
                        if(dis.at(k) < min && IOU_r.at(k) > IOU_threshold){
                            min = dis.at(k);
                            index = lost_index.at(k);
                        }
                    }
                    else{
                        if(dis.at(k) < min && IOU_r.at(k) >= 0.0){
                            min = dis.at(k);
                            index = lost_index.at(k);
                        }
                    }
                }
                qDebug() << "min dis" << min;

                int lost2track = sqrt(pow(total_fruit.at(index).h, 2) + pow(total_fruit.at(index).w, 2)) / 2;

                if(min < lost2track){   // 1. Lost -> Tracked
                    curr_vec.at(i).track_id = total_fruit.at(index).track_id;
                    total_fruit.at(index).history.pop_back();
                    total_fruit.at(index).history.append(2);
                    total_fruit.at(index).trajectory.append(curr_point);
                    total_fruit.at(index).frame_mat.append(save_frame);
                    total_fruit.at(index).width_height.append(w_h);
                    total_fruit.at(index).depth_hist.append(curr_vec.at(i).median_depth);
                    total_fruit.at(index).true_size_hist.append(curr_vec.at(i).true_size);
                    qDebug() << "1. Lost -> Tracked" << " ID:" << total_fruit.at(index).track_id;
                }
                else{   // 2. New Fruit
                    curr_vec.at(i).track_id = total_fruit.size() + 1;
                    for(int pp = 0 ; pp < curr_frame ; pp++){
                        curr_vec.at(i).history.append(0);    // Inactive
                    }
                    curr_vec.at(i).history.append(2);   // First Tracked
                    curr_vec.at(i).trajectory.append(curr_point);
                    curr_vec.at(i).frame_mat.append(save_frame);
                    curr_vec.at(i).width_height.append(w_h);
                    curr_vec.at(i).depth_hist.append(curr_vec.at(i).median_depth);
                    curr_vec.at(i).true_size_hist.append(curr_vec.at(i).true_size);
                    total_fruit.push_back(curr_vec.at(i));
                    qDebug() << "2. New Fruit" << " ID:" << curr_vec.at(i).track_id;
                }
            }
            else{   // There is no Lost fruit in total fruit --> New Fruit
                curr_vec.at(i).track_id = total_fruit.size() + 1;
                for(int pp = 0 ; pp < curr_frame ; pp++){
                    curr_vec.at(i).history.append(0);    // Inactive
                }
                curr_vec.at(i).history.append(2);   // First Tracked
                curr_vec.at(i).trajectory.append(curr_point);
                curr_vec.at(i).frame_mat.append(save_frame);
                curr_vec.at(i).width_height.append(w_h);
                curr_vec.at(i).depth_hist.append(curr_vec.at(i).median_depth);
                curr_vec.at(i).true_size_hist.append(curr_vec.at(i).true_size);
                total_fruit.push_back(curr_vec.at(i));
                qDebug() << "2. New Fruit (No Fruit Lost in total fruit)" << " ID:" << curr_vec.at(i).track_id;
            }
        }
    }
}

inline std::vector<bbox_t_history> bbox_t2bbox_t_history(std::vector<bbox_t> input){
    std::vector<bbox_t_history> output;
    for(int i = 0 ; i < input.size() ; i++){
        bbox_t_history temp;
        temp.h = input.at(i).h;
        temp.obj_id = input.at(i).obj_id;
        temp.prob = input.at(i).prob;
        temp.track_id = input.at(i).track_id;
        temp.w = input.at(i).w;
        temp.x = input.at(i).x;
        temp.y = input.at(i).y;
        output.push_back(temp);
    }
    return output;
}

#endif // FEATURE_FUNCTION_HPP

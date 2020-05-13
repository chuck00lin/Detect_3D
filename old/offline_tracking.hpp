#ifndef OFFLINE_TRACKING_HPP
#define OFFLINE_TRACKING_HPP

#include "detect_3d.hpp"

inline void save_track_result(QString track_result_path, int online, QList<int> false_alarm, std::string filename){

    QFile track_result(track_result_path);
    QTextStream out2(&track_result);
    if(track_result.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out2 << "======= Tracking Result =======\n";
        out2 << "Video: " << QString::fromUtf8(filename.c_str());
        out2 << "\nOnline-tracking\n";
        out2 << "total_fruit.size() = " << online;
        out2 << "\n\nOffline-tracking\nErase-ID\n";
        for(int i = 0 ; i < false_alarm.size() ; i++){
            out2 << false_alarm.at(i) << ", ";
        }
        out2 << "\ntotal_fruit.size() = " << online - false_alarm.size() << "\n\n";
    }
    track_result.close();
}

inline void save_histogram(QString save_path, QList<int> histogram_list, std::vector<std::pair<int, int>> max_min){
    QFile histogram(save_path);
    QTextStream out1(&histogram);
    if(histogram.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out1 << "======= Fruit size histogram ======== \n";
        for(int i = 0 ; i < histogram_list.size() ; i++){
            out1 << histogram_list.at(i) << ", ";
        }
    }
    out1 << "\nMax ID, size: " << max_min.at(0).first << ", " << max_min.at(0).second << "\n";
    out1 << "\nMax ID, size: " << max_min.at(1).first << ", " << max_min.at(1).second << "\n\n";
    histogram.close();
}

inline void save_ripening_stage(QString save_path, QList<double> stage_list){
    QFile ripening(save_path);
    QTextStream out(&ripening);
    if(ripening.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out << "======= Fruit ripening stage histogram ======= \n";
        for(int i = 0 ; i < stage_list.size() ; i++){
            out << stage_list.at(i) << ", ";
        }
    }
    ripening.close();
}

inline std::pair<int, int> max_fruit_frame_size(QList<cv::Point2d> total_fruit_width_height){
    int frame = 0, max = 0;
    for(int i = 0 ; i < total_fruit_width_height.size() ; i++){
        int size = total_fruit_width_height.at(i).x * total_fruit_width_height.at(i).y;
        if(size > max){
            max = size;
            frame = i;
        }
    }
    return std::make_pair(frame, max);
}

inline QList<int> Eliminate_false_alarm(std::vector<bbox_t_history>& total_fruit, int threshold){
    QList<int> erase_id;     // if < threshold frame == Tracked --> false alarm
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int count = 0;
        for(int j = total_fruit.at(i).history.size() - 1 ; j >= 0 ; j--){
            if(total_fruit.at(i).history.at(j) == 2)    count++;
            if(count > threshold)   break;
        }
        if(count <= threshold){
            erase_id.append(total_fruit.at(i).track_id);
            total_fruit.erase(std::remove_if(total_fruit.begin(), total_fruit.end(), [&](bbox_t_history &vector){
                                  return (vector.track_id == total_fruit.at(i).track_id);
                              }), total_fruit.end());
            i--;
        }
    }
    return erase_id;
}

inline QList<int> Eliminate_false_alarm(std::vector<bbox_t_history>& total_fruit, int threshold, int frame){
    QList<int> erase_id;     // if < threshold frame == Tracked --> false alarm
    for(int i = 0 ; i < total_fruit.size() ; i++){
        int count = 0;
        if(total_fruit.at(i).history.size() < frame)    continue;
        for(int j = total_fruit.at(i).history.size() - 1 ; j >= 0 ; j--){
            if(total_fruit.at(i).history.at(j) == 2)    count++;
            if(count > threshold)   break;
        }
        if(count <= threshold){
            erase_id.append(total_fruit.at(i).track_id);
            total_fruit.erase(std::remove_if(total_fruit.begin(), total_fruit.end(), [&](bbox_t_history &vector){
                                  return (vector.track_id == total_fruit.at(i).track_id);
                              }), total_fruit.end());
            i--;
        }
    }
    return erase_id;
}

inline std::pair<QList<int>, std::vector<std::pair<int, int>>> Fruit_size_histogram(std::vector<bbox_t_history>& total_fruit){
    QList<int> histogram;
    int min = 100000, max = 0;
    int min_ID = 0, max_ID = 0;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        std::pair<int, int> max_frame_size = max_fruit_frame_size(total_fruit.at(i).width_height);
        int size = max_frame_size.second;

        total_fruit.at(i).size = size;
        histogram.append(size);
        if(size < min){ min = size; min_ID = total_fruit.at(i).track_id;}
        if(size > max){ max = size; max_ID = total_fruit.at(i).track_id;}
    }
    std::vector<std::pair<int, int>> max_min;
    max_min.push_back(std::make_pair(max_ID, max));
    max_min.push_back(std::make_pair(min_ID, min));

    qDebug() << "Max ID, size: " << max_ID << ", " << max;
    qDebug() << "Min ID, size: " << min_ID << ", " << min;

    return std::make_pair(histogram, max_min);
}

inline double maturity(cv::Mat& mask, cv::Mat input){
    double maturity;
    int mask_pixel = 0, mature_pixel = 0;
    cv::Mat hsv = input.clone();
    cv::cvtColor(hsv, hsv, CV_BGR2HSV);

    QFile maturity_hsv("./depth_data/maturity_HSV_histogram.csv");
    QTextStream out(&maturity_hsv);
    maturity_hsv.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text);
    int stages[5] = {0, 0, 0, 0, 0};

    if(!mask.empty()){
        out << "\n";

        for(int i = 0 ; i < mask.rows ; i++){
            for(int j = 0 ; j < mask.cols ; j++){
                if(mask.at<uchar>(i, j) == 1 || mask.at<uchar>(i, j) == 3 || mask.at<uchar>(i, j) == 255){     // Foreground
                    mask_pixel++;

                    out << hsv.at<cv::Vec3b>(i, j)[0] << ",";
                    int value = hsv.at<cv::Vec3b>(i, j)[0];
                    if(value < 75 && value >= 45)   stages[0] += 1;
                    else if(value < 45 && value >= 20)   stages[1] += 1;
                    else if(value < 20 && value >= 10)   stages[2] += 1;
                    else if(value < 10 && value >= 5)   stages[3] += 1;
                    else if(value < 5 || value > 170)   stages[4] += 1;

                    // Mature condition
                    if((input.at<cv::Vec3b>(i, j)[2] >= 200 && input.at<cv::Vec3b>(i, j)[1] < 200 && input.at<cv::Vec3b>(i, j)[0] < 160)
                            || (input.at<cv::Vec3b>(i, j)[2] >= 165 && input.at<cv::Vec3b>(i, j)[1] < 80 && input.at<cv::Vec3b>(i, j)[0] < 80)
                            || (input.at<cv::Vec3b>(i, j)[2] >= 100 && input.at<cv::Vec3b>(i, j)[1] < 40 && input.at<cv::Vec3b>(i, j)[0] < 40)){
                        mature_pixel++;
                        mask.at<uchar>(i, j) = 100;
                    }
                    else{mask.at<uchar>(i, j) = 100;}
                }
                else    {mask.at<uchar>(i, j) = 0;}                             // Background
            }
        }
        int max_v = stages[0];
        double stage = 0.0;
        for(int s = 1 ; s < 5 ; s++){
            if(stages[s] > max_v) {
                stage = (double)s;
                max_v = stages[s];
            }
        }
        if(mature_pixel != 0 && mask_pixel != 0){
            maturity = stage;
        }
        else{
            maturity = 0.0;
        }
    }
    else{
        maturity = 0.0;
        qDebug() << "mask is empty";
    }
    maturity_hsv.close();

    return maturity;
}

inline void save_ripen_img(cv::Mat input, cv::Mat mask, std::string save_path, double stage, int ID){
    cv::Mat mask_result(input.rows, input.cols, CV_8UC3);
    for(int i = 0 ; i < mask.rows ; i++){
        for(int j = 0 ; j < mask.cols ; j++){
            mask_result.at<cv::Vec3b>(i, j)[0] = input.at<cv::Vec3b>(i, j)[0] * 0.6 + mask.at<uchar>(i, j) * 0.4;
            mask_result.at<cv::Vec3b>(i, j)[1] = input.at<cv::Vec3b>(i, j)[1] * 0.6 + mask.at<uchar>(i, j) * 0.4;
            mask_result.at<cv::Vec3b>(i, j)[2] = input.at<cv::Vec3b>(i, j)[2] * 0.6 + mask.at<uchar>(i, j) * 0.4;
        }
    }
    cv::putText(mask_result, "Fruit ID : " + std::to_string(ID), cv::Point2f(0, mask_result.rows - 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1.5);
    cv::putText(mask_result, "Ripening Stage : " + std::to_string(stage), cv::Point2f(0, mask_result.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1.5);

    cv::imwrite(save_path, mask_result);
}

inline double ripening_stage(cv::Mat input, cv::Point2f trajectory, cv::Point2f width_height, int ID){

    cv::Mat mask(input.rows, input.cols, CV_8U, cv::Scalar(0));
    cv::Mat bgModel, fgModel;
    int left_top_x = std::max((int)trajectory.x - (int)width_height.x / 2 - 10, 0);
    int left_top_y = std::max((int)trajectory.y - (int)width_height.y / 2 - 10, 0);
    int width = std::min((int)width_height.x + 20, (int)input.cols);
    int height = std::min((int)width_height.y + 20, (int)input.rows);
    cv::circle(mask, cv::Point(left_top_x + width / 2, left_top_y + height / 2), std::min(width / 2, height / 2) - 10, cv::Scalar(1), -1);
    cv::Rect rect(left_top_x, left_top_y, width, height);
    cv::grabCut(input, mask, rect, bgModel, fgModel, 10, cv::GC_INIT_WITH_RECT);

    int avai = 0;
    for(int i = 0 ; i < mask.rows ; i++){
        for(int j = 0 ; j < mask.cols ; j++){
            if(mask.at<uchar>(i, j) == 1 || mask.at<uchar>(i, j) == 3){
                avai++;
            }
        }
    }
//    qDebug() << "Foreground pixel: " << avai;

    if(avai == 0){      // If no foreground, often occur with green fruit --> use yolo result
        cv::Mat mask2 = cv::Mat::zeros(input.size(), CV_8U);
        cv::circle(mask2, cv::Point(left_top_x + width / 2, left_top_y + height / 2), std::min(width / 2, height / 2) - 10, cv::Scalar(255), -1);
        cv::Mat imagePart = cv::Mat::zeros(input.size(), input.type());
        mask = mask2.clone();
    }

    double maturity_per = maturity(mask, input);

    std::string save_path = "./tracking_frame/maturity_mask/fruit_" + std::to_string(ID) + ".png";
    save_ripen_img(input, mask, save_path, maturity_per, ID);

    return maturity_per;
}

inline QList<double> Fruit_ripening_stage(std::vector<bbox_t_history>& total_fruit){
    QList<double>   fruit_ripening_stage;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        std::pair<int, int> max_frame_size = max_fruit_frame_size(total_fruit.at(i).width_height);
        int n_frame = max_frame_size.first;

        cv::Mat input = total_fruit.at(i).frame_mat.at(n_frame).clone();
        double stage = ripening_stage(input, total_fruit.at(i).trajectory.at(n_frame), total_fruit.at(i).width_height.at(n_frame), total_fruit.at(i).track_id);
        total_fruit.at(i).maturity = stage;
        fruit_ripening_stage.append(stage);
    }
    return fruit_ripening_stage;
}

inline global_coor set_coordinate(bbox_t_history total_fruit, cv::Point2f point){

    global_coor coor;
    coor.global_point = point;
    coor.global_fruit_ID = total_fruit.track_id;
    coor.maturity = total_fruit.maturity;
    coor.size = total_fruit.size;

    return coor;
}

inline std::pair<int, int> near_origin_point(QList<cv::Point2f> trajectory_list, cv::Point2f origin, QList<unsigned int> history_list){
    double min = cv::norm(origin - trajectory_list.at(0));
    unsigned int trajectory_count = 1;
    for(int i = 1 ; i < trajectory_list.size() ; i++){
        double temp = cv::norm(origin - trajectory_list.at(i));
        if(temp < min){
            min = temp;
            trajectory_count = i;
        }
    }
    int count = 0;
    int history_index = 0;
    for(int j = 0 ; j < history_list.size() ; j++){
        if(history_list.at(j) == 2){
            count++;
            if(count == trajectory_count){
                history_index = j;
                break;
            }
        }
    }
    return std::make_pair(history_index, trajectory_count - 1);
}

inline std::pair<int, int> nearest_point(QList<double> depth_hist, QList<unsigned int> history_list){
    double min = depth_hist.at(0);
    unsigned int trajectory_count = 1;
    for(int i = 1 ; i < depth_hist.size() ; i++){
        double temp = depth_hist.at(i);
        if(temp < min){
            min = temp;
            trajectory_count = i;
        }
    }
    int count = 0;
    int history_index = 0;
    for(int j = 0 ; j < history_list.size() ; j++){
        if(history_list.at(j) == 2){        // If tracked
            count++;
            if(count == trajectory_count){
                history_index = j;
                break;
            }
        }
    }
    return std::make_pair(history_index, trajectory_count - 1);
}

inline QList<global_coor> Calculate_global_coordinate(std::vector<bbox_t_history> total_fruit, cv::Point2f& max_global, cv::Point2f& min_global, QList<cv::Mat> Homo_history, bool depth){
    QList<global_coor> global_coord;
    for(int i = 0 ; i < total_fruit.size() ; i++){
        // Calculate by nearest point  if depth == true
        if(depth){
            // cv::Point2f origin(1280 / 2, 720 / 2);  //(x, y)
            // std::pair<int, int> near = near_origin_point(total_fruit.at(i).trajectory, origin, total_fruit.at(i).history);
            std::pair<int, int> near = nearest_point(total_fruit.at(i).depth_hist, total_fruit.at(i).history);
            int history_index = near.first;
            int trajectory_index = near.second;

            // qDebug() << "history, trajectory index: " << history_index << ", " << trajectory_index;
            // qDebug() << "ID: " << total_fruit.at(i).track_id;
            cv::Point2f point = total_fruit.at(i).trajectory.at(trajectory_index);
            for(int h = 0 ; h < history_index ; h++){

                // qDebug() << "   point x, y " << point.x << point.y;
                cv::Point2f result = global_coordinate(Homo_history.at(h), point);
                // qDebug() << "   result x, y " << result.x << result.y;
                point.x = result.x;
                point.y = result.y;
            }
            global_coor coor;
            coor = set_coordinate(total_fruit.at(i), point);
            coor.nearest_trajectory_index = trajectory_index;
            global_coord.append(coor);
            if(point.x > max_global.x) max_global.x = point.x;
            if(point.y > max_global.y) max_global.y = point.y;
            if(point.x < min_global.x) min_global.x = point.x;
            if(point.y < min_global.y) min_global.y = point.y;
        }
        else{
            for(int j = 0 ; j < total_fruit.at(i).history.size() ; j++){
                if(total_fruit.at(i).history.at(j) == 2){   // First tracked point
                    cv::Point2f point = total_fruit.at(i).trajectory.at(0);
                    global_coor coor;
                    if(j == 0){         // If tracked in the first frame -> No need to calculate
                        coor = set_coordinate(total_fruit.at(i), point);
                    }
                    else{
                        for(int h = 0 ; h < j ; h++){
                            cv::Point2f result = global_coordinate(Homo_history.at(h), point);
                            point.x = result.x;
                            point.y = result.y;
                        }
                        coor = set_coordinate(total_fruit.at(i), point);
                    }
                    global_coord.append(coor);
                    if(point.x > max_global.x) max_global.x = point.x;
                    if(point.y > max_global.y) max_global.y = point.y;
                    if(point.x < min_global.x) min_global.x = point.x;
                    if(point.y < min_global.y) min_global.y = point.y;
                    break;
                }
            }
        }
    }
    return global_coord;
}

inline cv::Scalar set_maturity_color(double maturity){
    cv::Scalar color;
    cv::Scalar Maturity_color[5] = {cv::Scalar(9, 113, 59)
                                    , cv::Scalar(77, 190, 255)
                                    , cv::Scalar(9, 132, 255)
                                    , cv::Scalar(4, 96, 255)
                                    , cv::Scalar(0, 34, 255)};

    if((int)maturity == 0)  color = Maturity_color[0];
    else if((int)maturity == 1)  color = Maturity_color[1];
    else if((int)maturity == 2)  color = Maturity_color[2];
    else if((int)maturity == 3)  color = Maturity_color[3];
    else if((int)maturity == 4)  color = Maturity_color[4];

    return color;
}

inline int set_radius(int size, int size_bin[]){
    int radius;
    if(size >= size_bin[0] && size < size_bin[1])   radius = 4*2;
    else if(size >= size_bin[1] && size < size_bin[2])    radius = 8*2;
    else if(size >= size_bin[2] && size < size_bin[3])    radius = 12*2;
    else if(size >= size_bin[3] && size < size_bin[4])    radius = 16*2;
    else if(size >= size_bin[4] && size < size_bin[5])    radius = 20*2;
    else radius = 24*2;
    return radius;
}
#endif // OFFLINE_TRACKING_HPP

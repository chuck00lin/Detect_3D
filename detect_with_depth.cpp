#include "detect_with_depth.hpp"

//#include <cvhelpers.hpp>
#include <feature_function.hpp>
#include <offline_tracking.hpp>

/* ################ */
/*    YOLO: X, Y    */
/*  top-left corner */
/* ################ */

#define M_PI 3.14159265358979323846
std::mutex point_mutex;
using pixel = std::pair<int, int>;


detect_with_depth::detect_with_depth()
{

}

void detect_with_depth::get_bag_file(){
    QString name = QFileDialog::getOpenFileName(this, tr("Open .bag file"), "F://Fruit_harvest//Raw_data");
    qDebug()<<name;
    bag_filename = name.toStdString();
}

void detect_with_depth::initialize_realsense(){
    // Retrieve streams
    rs2::config cfg;

    rs2::context context;
    const rs2::playback playback = context.load_device(bag_filename);
    qDebug()<<"BUG1";
    const std::vector<rs2::sensor> sensors = playback.query_sensors();
    qDebug()<<"BUG2";
    for(const rs2::sensor& sensor : sensors){
        const std::vector<rs2::stream_profile> stream_profiles = sensor.get_stream_profiles();
        qDebug()<<"BUG3";
        for(const rs2::stream_profile& stream_profile : stream_profiles){
            cfg.enable_stream(stream_profile.stream_type(), stream_profile.stream_index());  
        }
    }


    // Start Pipeline
    cfg.enable_device_from_file(bag_filename);
    pipeline_profile = pipeline.start(cfg);
    pipeline_profile.get_device().as<rs2::playback>().set_real_time(false);
    qDebug()<<"BUG4";

    // Show enable streams
    qDebug() << "Enable streams: ";
    const std::vector<rs2::stream_profile> stream_profiles = pipeline_profile.get_streams();
    for(const rs2::stream_profile stream_profile : stream_profiles){
        qDebug() << QString::fromUtf8(stream_profile.stream_name().c_str());
    }
}

inline void detect_with_depth::initialize_detector(bool type){
    if(type){
        // 1 vs 1 model
        // cfg_file = "D://Fruit_harvest//Train_data//one_vs_one_model//tomato_JingYong//tomato_JingYong.cfg";
        // weights_file = "D://Fruit_harvest//Train_data//one_vs_one_model//tomato_JingYong//model//tomato_JingYong_4000.weights";
        // names_file = "D://Fruit_harvest//Train_data//one_vs_one_model//tomato_JingYong//tomato_JingYong.names";

        // 1st round data
        // cfg_file = "F://Fruit_harvest//Training//50%up//tomato_50up_v2.cfg";
        // weights_file = "F://Fruit_harvest//Training//50%up//model//tomato_50up_v2_15000.weights";
        // names_file = "F://Fruit_harvest//Training//50%up//tomato_50up.names";

        // 1st round data (modified)
        // cfg_file = "F://Fruit_harvest//Training//50%up//tomato_50up_v2_modified5_anchor.cfg";
        // weights_file = "F://Fruit_harvest//Training//50%up//model//tomato_50up_v2_modified_anchor//5anchor//save_weight//tomato_50up_v2_modified5_anchor_final.weights";
        // names_file = "F://Fruit_harvest//Training//50%up//tomato_50up.names";
        //

        //原本是這個
        //1st round data + internet + 2nd round data
        cfg_file = "F://Fruit_harvest//Training//50%up//tomato_50up_v2_add_internet_add_2nd_round_modified5_anchor.cfg";
        weights_file = "F://Fruit_harvest//Training//50%up//model//tomato_50up_v2_add_internet_add_2nd_round//tomato_50up_v2_add_internet_add_2nd_round_modified5_anchor_final.weights";
        names_file = "F://Fruit_harvest//Training//50%up//tomato_50up.names";


        //1st round data + 2nd round data +3nd round data
        //cfg_file = "F://Fruit_harvest//Training//50%up_v3//yolov3_tomato2.cfg";
        //weights_file = "F://Fruit_harvest//Training//50%up_v3//model//yolov3_tomato2_final.weights";
        //names_file = "F://Fruit_harvest//Training//50%up_v3//tomato2.names";

        // 1st round data + internet
        // cfg_file = "F://Fruit_harvest//Training//50%up//tomato_50up_v2_add_internet_modified5_anchor_aug_angle5.cfg";
        // weights_file = "F://Fruit_harvest//Training//50%up//model//tomato_50up_v2_add_internet//tomato_50up_v2_add_internet_modified5_anchor_aug_angle5_45000.weights";
        // names_file = "F://Fruit_harvest//Training//50%up//tomato_50up.names";

        // Only 100% model
        // cfg_file = "F://Fruit_harvest//Training//only100%//tomato_only100.cfg";
        // weights_file = "F://Fruit_harvest//Training//only100%//model//tomato_only100_15080.weights";
        // names_file = "F://Fruit_harvest//Training//only100%//tomato_only100.names";

    }
    else{
        // 1 vs 1 model
        cfg_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.cfg";
        weights_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//model//strawberry_JingYong_1000.weights";
        names_file = "D://Fruit_harvest//Train_data//one_vs_one_model//berry_JingYong//strawberry_JingYong.names";
    }
}

inline void detect_with_depth::Color(){
    color_frame = frameset.get_color_frame();
    if(!color_frame)    return;
    color_width = color_frame.as<rs2::video_frame>().get_width();
    color_height = color_frame.as<rs2::video_frame>().get_height();
}

inline void detect_with_depth::Depth(){
    depth_frame = frameset.get_depth_frame();
    if(!depth_frame)    return;
    depth_width = depth_frame.as<rs2::video_frame>().get_width();
    depth_height = depth_frame.as<rs2::video_frame>().get_height();
}

cv::Mat draw_detect_point(std::vector<bbox_t> result_vector, cv::Mat input, cv::Scalar fruit_point_color, bool bbox){
    cv::Mat output = input.clone();
    for(int i = 0 ; i < result_vector.size() ; i++){
        cv::Point2f temp((float)result_vector[i].x + (float)result_vector[i].w / 2, (float)result_vector[i].y + (float)result_vector[i].h / 2);
        cv::circle(output, temp, 5, fruit_point_color, -1);
        cv::Point2f temp1((float)result_vector[i].x, (float)result_vector[i].y);
        qDebug() << "x, y " << ((int)temp1.x) << ", " << ((int)temp1.y);
        //        cv::putText(output, "x, y " + std::to_string((int)temp1.x) + ", " + std::to_string((int)temp1.y)
        //                    , temp, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, fruit_point_color, 1);
        if(bbox){
            cv::rectangle(output, cv::Point2f((float)result_vector[i].x, (float)result_vector[i].y)
                          , cv::Point2f((float)result_vector[i].x + (float)result_vector[i].w, (float)result_vector[i].y + (float)result_vector[i].h)
                          , fruit_point_color, 1);
        }
    }
    return output;
}

cv::Mat draw_detect_point(std::vector<bbox_t_history> result_vector, cv::Mat input, cv::Scalar fruit_point_color, bool bbox){
    cv::Mat output = input.clone();
    for(int i = 0 ; i < result_vector.size() ; i++){
        cv::Point2f temp((float)result_vector[i].x + (float)result_vector[i].w / 2, (float)result_vector[i].y + (float)result_vector[i].h / 2);
        cv::circle(output, temp, 5, fruit_point_color, -1);
        //        cv::ellipse(output, temp, cv::Size(20, 30), 0, 0, 360, cv::Scalar(0, 0, 0), -1);
        cv::Point2f temp1((float)result_vector[i].x, (float)result_vector[i].y);
        //        cv::putText(output, "x, y" + std::to_string((int)temp1.x) + ", " + std::to_string((int)temp1.y)
        //                    , temp, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, fruit_point_color, 1);
        if(bbox){
            cv::rectangle(output, cv::Point2f((float)result_vector[i].x, (float)result_vector[i].y)
                          , cv::Point2f((float)result_vector[i].x + (float)result_vector[i].w, (float)result_vector[i].y + (float)result_vector[i].h)
                          , fruit_point_color, 1);
        }
    }
    return output;
}

inline double mean_distance(std::pair<int, int> window_size, rs2::depth_frame depth_info, bbox_t_history vector){
    double distance = 0.0;
    int zero_n = 0;
    for(int i = 0 ; i < window_size.first ; i++){
        for(int j = 0 ; j < window_size.first ; j++){
            int x = vector.x + i;
            int y = vector.y + j;
            if(x < depth_info.get_width() && y < depth_info.get_height()){
                if(depth_info.get_distance(x, y) == 0)  zero_n++;
                else distance += depth_info.get_distance(x, y);
            }

        }
    }
    if((window_size.first * window_size.second - zero_n) != 0){
        return distance / (window_size.first * window_size.second - zero_n);
    }
    else return 0.0;
}

double frame_mean_distance(rs2::depth_frame depth_info, int depth_height, int depth_width){
    double depth = 0.0;
    int count = 0;
    for(int i = 0 ; i < depth_height ; i++){
        for(int j = 0 ; j < depth_width ; j++){
            double d = depth_info.get_distance(j, i);
            if(d >= 0.27 && d <= 1.8 ){
                depth += d;
                count++;
            }
        }
    }
    qDebug() << "depth , count " << depth << count;
    depth /= count;
    return depth;
}

double cal_SD(std::vector<double> distance, double mean){
    double sum = 0.0;
    for(size_t i = 0 ; i < distance.size() ; i++){
        sum += pow(distance.at(i) - mean, 2);
    }
    return sqrt(sum /= distance.size());
}

std::pair<double, double> median_distance(std::pair<int, int> window_size, rs2::depth_frame depth_info, std::pair<int, int> x_y){
    // RETURN: median distance, standar deviation
    std::vector<double> distances;
    double mean = 0.0;
    for(int i = 0 ; i < window_size.first ; i++){        //.first: width (col) (x)
        for(int j = 0 ; j < window_size.second ; j++){   //.second: height (row) (y)
            int x = x_y.first + i;      // .first: x
            int y = x_y.second + j;     // .second: y
            if(x < depth_info.get_width() && x >= 0 && y < depth_info.get_height() && y >= 0){
                double d = depth_info.get_distance(x, y);
                if(d != 0){
                    distances.push_back(d);
                }
                mean += d;
            }
        }
    }
    mean /= window_size.first * window_size.second;
    double SD = cal_SD(distances, mean);
    if(distances.size() > 0){
        std::sort(distances.begin(), distances.end());
        return std::make_pair(distances.at(distances.size() / 2), SD);
    }
    else{
        qDebug() << "all zero";
        return std::make_pair(0.0, 0.0);
    }
}

cv::Mat drawHistImg(cv::Mat cropped){
    int histSize = 256;
    float range[] = {0, 255} ;
    const float* histRange = {range};
    cv::Mat histImg_H, histImg_S, histImg_V;

    std::vector<cv::Mat> channels(3);
    cv::split(cropped, channels);

    cv::calcHist(&channels[0], 1, 0, cv::Mat (), histImg_H, 1, &histSize, &histRange, true, false);
    cv::calcHist(&channels[1], 1, 0, cv::Mat (), histImg_S, 1, &histSize, &histRange, true, false);
    cv::calcHist(&channels[2], 1, 0, cv::Mat (), histImg_V, 1, &histSize, &histRange, true, false);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    cv::normalize(histImg_H, histImg_H, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(histImg_S, histImg_S, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(histImg_V, histImg_V, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    qDebug() << "ROund: " << cvRound(histImg_H.at<float>(398));
    for( int i = 1; i < histSize; i++ ){
        cv::line( histImage, cv::Point( bin_w*(i-1), std::max(hist_h - cvRound(histImg_H.at<float>(i-1)), 0) ),
                  cv::Point( bin_w*(i), std::max(hist_h - cvRound(histImg_H.at<float>(i)), 0) ),
                  cv::Scalar( 255, 0, 0), 2, 8, 0  );
        cv::line( histImage, cv::Point( bin_w*(i-1), std::max(hist_h - cvRound(histImg_S.at<float>(i-1)), 0) ),
                  cv::Point( bin_w*(i), std::max(hist_h - cvRound(histImg_S.at<float>(i)), 0) ),
                  cv::Scalar( 0, 255, 0), 2, 8, 0  );
        cv::line( histImage, cv::Point( bin_w*(i-1), std::max(hist_h - cvRound(histImg_V.at<float>(i-1)), 0) ),
                  cv::Point( bin_w*(i), std::max(hist_h - cvRound(histImg_V.at<float>(i)), 0) ),
                  cv::Scalar( 0, 0, 255), 2, 8, 0  );
    }

    return histImage;
}

float dist_3d(const rs2::depth_frame frame, pixel u, pixel v){
    float upixel[2];
    float upoint[3];
    float vpixel[2];
    float vpoint[3];

    upixel[0] = u.first;
    upixel[1] = u.second;
    vpixel[0] = v.first;
    vpixel[1] = v.second;

    auto udist = frame.get_distance(upixel[0], upixel[1]);
    auto vdist = frame.get_distance(vpixel[0], vpixel[1]);
    //    qDebug() << "UDIST: " << udist;
    //    qDebug() << "VDIST: " << vdist;


    rs2_intrinsics intr = frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
    rs2_deproject_pixel_to_point(upoint, &intr, upixel, udist);
    rs2_deproject_pixel_to_point(vpoint, &intr, vpixel, vdist);

    //    if(intr.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY) qDebug() << "BROWN contrady";
    //    else    qDebug() << "NO DISTORTION";

    return sqrt(pow(upoint[0] - vpoint[0], 2) +
            pow(upoint[1] - vpoint[1], 2) +
            pow(upoint[2] - vpoint[2], 2));
}

std::pair<float, float> fruit_true_size(rs2::depth_frame depth_info, int x, int y, int w , int h){

    int img_w = depth_info.get_width();
    int img_h = depth_info.get_height();

    static int LRxTBy[4];
    LRxTBy[0] = std::max(std::min(x, img_w-1), 0);
    LRxTBy[1] = std::max(std::min(x, img_w-1), 0);
    LRxTBy[2] = std::max(std::min(y, img_h-1), 0);
    LRxTBy[3] = std::max(std::min(y, img_h-1), 0);

    //        qDebug() << "Origin L_x R_x T_y B_y w h" << LRxTBy[0] << LRxTBy[1]+w << LRxTBy[2] << LRxTBy[3]+h << w << h;
    float left = depth_info.get_distance(std::min(LRxTBy[0], img_w-1), std::min(y+h/2, img_h-1));
    float right = depth_info.get_distance(std::min(LRxTBy[1]+w, img_w-1), std::min(y+h/2, img_h-1));
    float top = depth_info.get_distance(std::min(x+w/2, img_w-1), std::min(LRxTBy[2], img_h-1));
    float bottom = depth_info.get_distance(std::min(x+w/2, img_w-1), std::min(LRxTBy[3]+h, img_h-1));
    //    qDebug() << "Initialize";
    while(left == 0 && LRxTBy[0] < img_w){
        LRxTBy[0]++;
        left = depth_info.get_distance(std::min(LRxTBy[0], img_w-1), std::min(y+h/2, img_h-1));
        //        qDebug() << left;
    }
    //    qDebug() << "LEFT";
    while(right == 0 && LRxTBy[1] > 0){
        LRxTBy[1]--;
        right = depth_info.get_distance(std::min(LRxTBy[1]+w, img_w-1), std::min(y+h/2, img_h-1));
        //        qDebug() << right;
    }
    //    qDebug() << "LEFT";
    while(top == 0 && LRxTBy[2] < img_h){
        LRxTBy[2]++;
        top = depth_info.get_distance(std::min(x+w/2, img_w-1), std::min(LRxTBy[2], img_h-1));
        //        qDebug() << top;
    }
    //    qDebug() << "LEFT";
    while(bottom == 0 && LRxTBy[3] > 0){
        LRxTBy[3]--;
        bottom = depth_info.get_distance(std::min(x+w/2, img_w-1), std::min(LRxTBy[3]+h, img_h-1));
        //        qDebug() << bottom;
    }
    //    qDebug() << "Latter L_x R_x T_y B_y w h" << LRxTBy[0] << LRxTBy[1]+w << LRxTBy[2] << LRxTBy[3]+h << w << h;

    pixel wfrom_pixel{LRxTBy[0], y+h/2};
    pixel wto_pixel{std::min(LRxTBy[1]+w, img_w-1), y+h/2};
    float wair_dist = dist_3d(depth_info, wfrom_pixel, wto_pixel);
    pixel hfrom_pixel{x+w/2, LRxTBy[2]};
    pixel hto_pixel{x+w/2, std::min(LRxTBy[3]+h, img_h-1)};
    float hair_dist = dist_3d(depth_info, hfrom_pixel, hto_pixel);
    std::pair<float, float> true_size(wair_dist, hair_dist);

    return true_size;
}

void Kick_fruit_out_of_boundary(std::vector<bbox_t_history> &result_vector, int boundary, cv::Size img_size
                                , rs2::depth_frame depth_info, std::pair<double, double> depth_threshold, cv::Mat input_img, int frame){
    double SD_threshold = 0.3;

    result_vector.erase(std::remove_if(result_vector.begin(), result_vector.end(), [&](bbox_t_history &vector){
                            return ((vector.x + vector.w / 2) < boundary) ||
                            ((vector.x + vector.w / 2) > (img_size.width - boundary)) ||
                            ((vector.y + vector.h / 2) < boundary) ||
                            ((vector.y + vector.h / 2) > (img_size.height - boundary));
                        }), result_vector.end());
    for(int i = 0 ; i < result_vector.size() ; i++){
        // Calculate Median distance
        int x = result_vector.at(i).x;
        int y = result_vector.at(i).y;
        int w = result_vector.at(i).w;
        int h = result_vector.at(i).h;
        std::pair<int, int> window_size(w, h);
        std::pair<int, int> x_y(x, y);
        std::pair<double, double> median_SD = median_distance(window_size, depth_info, x_y);
        //        double mean_d = mean_distance(window_size, depth_info, result_vector.at(i));
        double median_d = median_SD.first;
        double SD = median_SD.second;       // if SD is too big --> might be a leaf
        //        qDebug() << x << y << w << h << "Median distance, SD: " << median_d << SD << "Mean distance: " << mean_d;
        //        qDebug() << x << y << w << h << "SD: " << SD;
        //        qDebug() << "depth width, height" << depth_info.get_width() << depth_info.get_height();

        result_vector.at(i).median_depth = median_d;

        // Calculate True length of fruit
        std::pair<float, float> true_size;
        true_size = fruit_true_size(depth_info, x, y, w, h);   // .first: width .second: height
        double wair_dist = true_size.first;
        double hair_dist = true_size.second;
        //                qDebug() << "width height air_dist: " << wair_dist << hair_dist;
        result_vector.at(i).true_size = true_size;
        float wh_ratio = wair_dist / hair_dist;
        //        qDebug() << "wh_ratio" << wh_ratio;

        if(median_d < depth_threshold.first || median_d > depth_threshold.second || SD > SD_threshold)  // .first: lower bound of depth ; .second: upper bound of depth
            //        if(median_d < depth_threshold.first || median_d > depth_threshold.second)  // .first: lower bound of depth ; .second: upper bound of depth
            result_vector.at(i).prob = -1.0;
    }
    result_vector.erase(std::remove_if(result_vector.begin(), result_vector.end(), [&](bbox_t_history &vector){
                            return (vector.prob < 0.0);
                        }), result_vector.end());
}

void draw_ID(std::vector<bbox_t_history> vector, cv::Mat &draw_mat, int num_total){
    for(int i = 0 ; i < vector.size() ; i++){
        cv::putText(draw_mat, "ID:" + std::to_string(vector.at(i).track_id)
                    ,cv::Point2f((float)vector[i].x + (float)vector[i].w / 2 - 30, (float)vector[i].y + (float)vector[i].h / 2 + 20)
                    , cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
    }
    cv::putText(draw_mat, "Total Fruit : " + std::to_string(num_total), cv::Point2f(0, draw_mat.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
}

inline void save_history(std::vector<bbox_t_history> total_fruit, int num_frame){
    QFile history("./depth_data/history.csv");
    QTextStream out(&history);
    if(history.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text)){
        out << "Frame : " << num_frame << "\n";
        for(int i = 0 ; i < total_fruit.size() ; i++){
            out << "ID: " << total_fruit.at(i).track_id << "  ";
            for(int j = 0 ; j < total_fruit.at(i).history.size() ; j++){
                out << total_fruit.at(i).history.at(j) << " ";
            }
            out << "\n";
        }
    }
    history.close();
}

void detect_with_depth::set_saveiou(int arg1){
    if(arg1 == Qt::Checked) save_IOU = true;
    else    save_IOU = false;
    qDebug() << "saveIOU clicked" << save_IOU;
}

std::pair<float, float> pixel_size_ratio(bbox_t_history total_fruit){
    std::pair<float, float> true_size = total_fruit.true_size_hist.at(0);
    cv::Point2d W_H = total_fruit.width_height.at(total_fruit.width_height.size()-1);
    //    qDebug() << true_size.first << true_size.second << W_H.x << W_H.y;
    std::pair<float, float> ratio(W_H.x / true_size.first, W_H.y / true_size.second);
    return ratio;   // PIXEL / SIZE(m)
}

std::pair<double, double> moving_degree(cv::Point2f source, cv::Mat last_homo, cv::Mat curr_homo){      // Degree, norm
    cv::Point2f last_p = get_tracked_point(last_homo, source);
    cv::Point2f curr_p = get_tracked_point(curr_homo, source);
    cv::Point2f last_vector(last_p.x - source.x, last_p.y - source.y);
    cv::Point2f curr_vector(curr_p.x - source.x, curr_p.y - source.y);
    double last_norm = cv::norm(last_vector - cv::Point2f(0.0, 0.0));
    double curr_norm = cv::norm(curr_vector - cv::Point2f(0.0, 0.0));

    if(last_norm == 0 || curr_norm == 0)
        return std::make_pair(0.0, curr_norm);
    double degree = (180/M_PI) * acos((last_vector.x * curr_vector.x + last_vector.y * curr_vector.y) / (last_norm*curr_norm));
    qDebug() << "Last vector: " << last_vector.x << " " << last_vector.y;
    qDebug() << "Curr vector: " << curr_vector.x << " " << curr_vector.y;
    qDebug() << "Curr length: " << curr_norm;
    qDebug() << "Degree: " << degree;
    return std::make_pair(degree, curr_norm);
}

cv::Mat preprocess(cv::Mat input, rs2::depth_frame depth_info, std::pair<double, double> depth_threshold){
    double alpha = 1.0;
    cv::Mat output = input.clone();
    for(int i = 0 ; i < input.rows ; i++){
        for(int j = 0 ; j < input.cols ; j++){
            double dist = depth_info.get_distance(j, i);
            if(dist < depth_threshold.first || dist > depth_threshold.second){
                //                qDebug() << (input.at<cv::Vec3b>(i, j)[0] * alpha) << (input.at<cv::Vec3b>(i, j)[1] * alpha) << (input.at<cv::Vec3b>(i, j)[2] * alpha);
                output.at<cv::Vec3b>(i, j)[0] = (input.at<cv::Vec3b>(i, j)[0] * alpha);
                output.at<cv::Vec3b>(i, j)[1] = (input.at<cv::Vec3b>(i, j)[1] * alpha);
                output.at<cv::Vec3b>(i, j)[2] = (input.at<cv::Vec3b>(i, j)[2] * alpha);
            }
        }
    }
    cv::imshow("Preprocess", output);
    return output;
}

cv::Mat postprocess(cv::Mat input, std::vector<bbox_t> bbox, cv::Mat src){
    int boundary = 20;
    cv::Mat output = input.clone();
    for(int i = 0 ; i < input.rows ; i++){
        for(int j = 0 ; j < input.cols ; j++){
            for(int b = 0 ; b < bbox.size() ; b++){
                int x = bbox.at(b).x;
                int y = bbox.at(b).y;
                int w = bbox.at(b).w;
                int h = bbox.at(b).h;
                if(((j >= (x - boundary) && j <= (x + w + boundary)) && (i >= y - boundary && i <= (y + h + boundary)))){
                    output.at<cv::Vec3b>(i, j)[0] = src.at<cv::Vec3b>(i, j)[0];
                    output.at<cv::Vec3b>(i, j)[1] = src.at<cv::Vec3b>(i, j)[1];
                    output.at<cv::Vec3b>(i, j)[2] = src.at<cv::Vec3b>(i, j)[2];
                }
            }
        }
    }
    cv::imshow("Postprocess", output);
    return output;
}

void detect_with_depth::run(){
    bool tomato = true;
    get_bag_file();
    initialize_realsense();
    initialize_detector(tomato);  //true: tomato ; false: strawberry
    Detector detector(cfg_file, weights_file);

    cv::Scalar fruit_point_color(0, 0, 255);
    cv::Scalar tracked_point_color(255, 0, 0);
    cv::Scalar boundary_color(255, 255, 0);

    int threshold = 35;     // Previous tracked fruit vs current fruit distance threshold //單位是pixel嗎
    int lost_track_threshold = 45;      // Lost fruit return to track fruit threshold
    double detect_threshold = 0.5;
    //    double detect_threshold = 0.07;
    int img_boundary = 40;
    bool draw_bbox = true;  // Draw detection bounding box or not
    std::pair<double, double> depth_threshold(0.30, 0.9);  // .first: lower bound of depth ;  .second: upper bound of depth
    save_IOU = false;
    int prev_fruit_count = 0;
    int prev_fruit_count_threshold = 50;    // If accumulate frame with no fruit above X, then it really didn't detect fruit
    int FA_frame = 50;  // Eliminate false alarm per 50 frame



    srand(time(NULL));
    cv::Mat frame0, frame1, draw_mat, input;
    rs2::frame depth_frame0, depth_frame1;
    rs2::align align_to(RS2_STREAM_COLOR);
    int i = 0;
    while(i < 2){
        frameset = pipeline.wait_for_frames();
        frameset = align_to.process(frameset);
        Color();
        Depth();
        draw_color();
        draw_depth();
        showColor();
        showDepth();
        draw_mat = color_mat.clone();
        input = color_mat.clone();
        if(i == 0){
            frame0 = color_mat.clone();
            depth_frame0 = depth_frame;
        }
        if(i == 1){
            frame1 = color_mat.clone();
            depth_frame1 = depth_frame;
        }
        i++;
    }

    // Fruit detection in frame 0, frame 1
    std::vector<bbox_t> result_vec0 = detector.detect(frame0, detect_threshold);
    std::vector<bbox_t> result_vec1 = detector.detect(frame1, detect_threshold);

    draw_mat = draw_detect_point(result_vec0, frame0, fruit_point_color, draw_bbox);
    cv::rectangle(draw_mat, cv::Point2f(img_boundary, img_boundary), cv::Point2f(color_width - img_boundary, color_height - img_boundary),
                  boundary_color, 1);
    cv::imshow("BEFORE", draw_mat);
    draw_mat = draw_detect_point(result_vec1, frame1, fruit_point_color, draw_bbox);


    cv::Size img_size(color_width, color_height);
    std::vector<bbox_t_history> tranform0 = bbox_t2bbox_t_history(result_vec0);
    std::vector<bbox_t_history> tranform1 = bbox_t2bbox_t_history(result_vec1);

    Kick_fruit_out_of_boundary(tranform0, img_boundary, img_size, depth_frame0, depth_threshold, frame0, 0);
    Kick_fruit_out_of_boundary(tranform1, img_boundary, img_size, depth_frame1, depth_threshold, frame1, 1);

    // Draw boundary = 40
    draw_mat = draw_detect_point(tranform0, frame0, fruit_point_color, draw_bbox);
    cv::rectangle(draw_mat, cv::Point2f(img_boundary, img_boundary), cv::Point2f(color_width - img_boundary, color_height - img_boundary),
                  boundary_color, 1);
    cv::imshow("AFTER", draw_mat);

    result_vec0.clear();
    result_vec1.clear();
    std::vector<bbox_t_history> total_fruit;
    for(int i = 0 ; i < tranform0.size() ; i++){
        tranform0.at(i).track_id = i + 1;
        total_fruit.push_back(tranform0.at(i));
        cv::Point2f trajectory((float)tranform0[i].x + (float)tranform0[i].w / 2, (float)tranform0[i].y + (float)tranform0[i].h / 2);
        cv::Point2f wh((float)tranform0[i].w, (float)tranform0[i].h);
        total_fruit.at(i).trajectory.append(trajectory);
        total_fruit.at(i).history.append(2);
        total_fruit.at(i).width_height.append(wh);
        total_fruit.at(i).frame_mat.append(frame0);
        total_fruit.at(i).depth_hist.append(tranform0.at(i).median_depth);
        total_fruit.at(i).true_size_hist.append(tranform0.at(i).true_size);
        cv::putText(draw_mat, "ID:" + std::to_string(tranform0.at(i).track_id), cv::Point2f(tranform0.at(i).x - 30, tranform0.at(i).y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1.5);
    }

    // Current / Previous mat fruit
    QList<cv::Point2f> fruit0;
    for(int i = 0 ; i < tranform0.size() ; i++){
        cv::Point2f temp((float)tranform0[i].x + (float)tranform0[i].w / 2, (float)tranform0[i].y + (float)tranform0[i].h / 2);
        fruit0.append(temp);
    }

    // Feature operation in frame 0
    QList<float> avg_point_dist_hist;
    std::vector<cv::Point2f> point0, point1;
    float avg_point_dist;
    std::vector<uchar> status;
    avg_point_dist = featureTracking_GPU(frame0, frame1, point0, point1, status, depth_frame0, true);
    avg_point_dist_hist.append(avg_point_dist);
    qDebug() << "avg_point_dist: " << avg_point_dist;

    // Calculate tracked previous point
    QList<cv::Mat>  Homo_history;
    QList<double> mean_depth_diff;
    double mean_depth0 = frame_mean_distance(depth_frame0, depth_height, depth_width);
    double mean_depth1 = frame_mean_distance(depth_frame1, depth_height, depth_width);
    mean_depth_diff.append(mean_depth1 - mean_depth0);

    cv::Mat homo_matrix = cv::findHomography(point0, point1);
    Homo_history.append(homo_matrix);
    QList<int> threshold_vec;
    for(int i = 0 ; i < tranform0.size() ; i++){
        int th = sqrt(pow(tranform0.at(i).h, 2) + pow(tranform0.at(i).w, 2)) / 2;
        threshold_vec.append(th);
    }
    for(int i = 0 ; i < fruit0.size() ; i++){
        cv::Point2f temp_point = get_tracked_point(homo_matrix, fruit0.at(i));
        cv::circle(draw_mat, temp_point, 5, tracked_point_color, -1);
        cv::circle(draw_mat, temp_point, threshold_vec.at(i), tracked_point_color, 1);
    }

    // Set id with tracking result
    bool previous_fruit = true;
    if(tranform0.size() == 0)    previous_fruit = false;
    cv::Mat check_mat = input.clone();
    set_ID_fast(total_fruit, tranform0, tranform1, Homo_history, mean_depth_diff, previous_fruit, threshold_vec, lost_track_threshold, avg_point_dist_hist, check_mat, check_mat, save_IOU, true);
    threshold_vec.clear();
    cv::imwrite("./depth_data/1_frame.jpg", draw_mat);

    // Define previous and current
    cv::Mat prev_mat, curr_mat;
    std::vector<cv::Point2f> prev_point, curr_point;
    std::vector<bbox_t_history> prev_vec, curr_vec;
    double prev_mean_depth, curr_mean_depth;

    prev_mat = frame1.clone();
    prev_point = point1;
    prev_vec = tranform1;
    prev_mean_depth = frame_mean_distance(depth_frame1, depth_height, depth_width);

    qDebug() << "================ END PRE-RUNNING ==================";

    // Preprocessing setting
    rs2::decimation_filter dec;
    dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    rs2::disparity_transform depth2disparity;
    rs2::disparity_transform disparity2depth(false);
    rs2::spatial_filter spat;
    spat.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    spat.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5);
    spat.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
    //    spat.set_option(RS2_OPTION_HOLES_FILL, 5);
    rs2::temporal_filter temp;


    // Tracking start with second frame in the video
    int frame = 0;
    uint64_t last_position = pipeline_profile.get_device().as<rs2::playback>().get_position();

    while(true){
//        if(frame == 30)    break;
        if(pipeline.poll_for_frames(&frameset)){

            QElapsedTimer timer;
            timer.start();

            frameset = align_to.process(frameset);
            // frameset = frameset.apply_filter(dec);    // After applying the decimal filter, the img resolution will decrease
            frameset = frameset.apply_filter(depth2disparity);
            frameset = frameset.apply_filter(spat);
            frameset = frameset.apply_filter(temp);
            frameset = frameset.apply_filter(disparity2depth);

            Color();
            Depth();
            draw_color();
            draw_depth();

            showDepth();

            rs2::frame depth_frame_;
            depth_frame_ = depth_frame;
            cv::Mat draw_mat = color_mat.clone();
            cv::Mat maturity_mat = color_mat.clone();
            if(draw_mat.empty())   break;

            if((frame > 2) && (frame%1 == 0)){

                threshold_vec.clear();
                curr_mat = color_mat.clone();

                // -------- Save img: Row data, Training, Testing imgs ------- //
                 cv::imwrite("./Row_frame/Tomato/0815_up_06/0815_up_06_" + std::to_string(frame) + ".png", curr_mat);
                // if(frame % 20 == 0) cv::imwrite("./training_img/0411_26_" + std::to_string(frame) + ".png", curr_mat);
                // if(frame % 35 == 0 && frame % 20 != 0) cv::imwrite("./testing_img/tomato/62/62_" + std::to_string(frame) + ".png", curr_mat);

                std::vector<uchar> status;
                avg_point_dist = featureTracking_GPU(prev_mat, curr_mat, prev_point, curr_point, status, depth_frame, true);

                // cv::Mat gt = preprocess(color_mat, depth_frame, depth_threshold);

                std::vector<bbox_t> temp;
                if(frame % 5 == 0){
                    temp = detector.detect(curr_mat, detect_threshold);
                    // cv::Mat pp = postprocess(gt, temp, color_mat);
                    // temp = detector.detect(pp, detect_threshold);
                }
                curr_vec = bbox_t2bbox_t_history(temp);
                temp.clear();

                qDebug() << "frame: " << frame;
                qDebug() << "prev_point, curr_point.size" << prev_point.size() << ", " << curr_point.size();


                avg_point_dist_hist.append(avg_point_dist);

                // Kick fruit out of boundary/too far and set fruit depth
                Kick_fruit_out_of_boundary(curr_vec, img_boundary, img_size, depth_frame, depth_threshold, curr_mat, frame);
                curr_mean_depth = frame_mean_distance(depth_frame, depth_height, depth_width);  // Frame mean depth


                // Current / Previous mat fruit
                QList<cv::Point2f> prev_fruit;
                draw_mat = draw_detect_point(curr_vec, draw_mat, fruit_point_color, draw_bbox);
                for(int i = 0 ; i < prev_vec.size() ; i++){
                    cv::Point2f temp((float)prev_vec[i].x + (float)prev_vec[i].w / 2, (float)prev_vec[i].y + (float)prev_vec[i].h / 2);
                    prev_fruit.append(temp);
                    int th = sqrt(pow(prev_vec.at(i).h, 2) + pow(prev_vec.at(i).w, 2)) / 2;
                    threshold_vec.append(th);
                }

                cv::rectangle(draw_mat, cv::Point2f(img_boundary, img_boundary), cv::Point2f(draw_mat.cols - img_boundary, draw_mat.rows - img_boundary),
                              boundary_color, 1);


                // Calculate and Draw tracked previous point
                qDebug() << "estimate rigid transform";
                cv::Mat rigid_trans = cv::estimateRigidTransform(prev_point, curr_point, false);
                Homo_history.append(rigid_trans);

                mean_depth_diff.append(prev_mean_depth - curr_mean_depth);
                qDebug() << "## prev depth, curr depth, diff" << prev_mean_depth << " " << curr_mean_depth << " " << prev_mean_depth - curr_mean_depth;

                rs2::depth_frame depth_FF = depth_frame;
                for(int i = 0 ; i < prev_fruit.size() ; i++){

                    // --------- Calculate distance ratio ------ //
                    // float prev_fruit_dist = depth_FF.get_distance((int)prev_fruit.at(i).x, (int)prev_fruit.at(i).y);
                    // float ratio = prev_fruit_dist / avg_point_dist; // (m/m)
                    float ratio = 1.0; // (m/m)
                    // ----------------------------------------- //

                    cv::Point2f temp_point = get_tracked_point(rigid_trans, prev_fruit.at(i));

                    float x_diff = temp_point.x - prev_fruit.at(i).x;
                    float y_diff = temp_point.y - prev_fruit.at(i).y;
                    temp_point.x = x_diff * ratio + prev_fruit.at(i).x;
                    temp_point.y = y_diff * ratio + prev_fruit.at(i).y;

                    cv::circle(draw_mat, temp_point, 5, tracked_point_color, -1);                                   // Blue Dot: predicted point
                    cv::circle(draw_mat, temp_point, std::min(threshold_vec.at(i), 50), tracked_point_color, 1);    // Blue Circle: threshold

                    if(draw_bbox){
                        cv::rectangle(draw_mat, cv::Point2f((float)prev_vec[i].x, (float)prev_vec[i].y)
                                      , cv::Point2f((float)prev_vec[i].x + (float)prev_vec[i].w, (float)prev_vec[i].y + (float)prev_vec[i].h)
                                      , tracked_point_color, 1);
                    }
                }

                // --------- Check if there is any fruit in the following `prev_fruit_count_threshold = 50` frames ------- //
                bool previous_fruit = true;
                if(prev_vec.size() == 0){
                    prev_fruit_count++;
                }
                else{
                    prev_fruit_count = 0;
                }
                if(prev_fruit_count > prev_fruit_count_threshold) previous_fruit = false;
                // ---------------------------------------------------------------------------------------------------- //

                // Set ID with tracking result
                qDebug() << "Start Set ID";
                set_ID_fast(total_fruit, prev_vec, curr_vec, Homo_history, mean_depth_diff, previous_fruit, threshold_vec, lost_track_threshold, avg_point_dist_hist, draw_mat, maturity_mat, save_IOU, true);

                // Save history and Draw ID
                save_history(total_fruit, frame);
                draw_ID(curr_vec, draw_mat, total_fruit.size());
                cv::imwrite("./depth_data/" + std::to_string(frame) + ".jpg", draw_mat);

                prev_mat = curr_mat.clone();
                prev_point = curr_point;
                prev_vec = curr_vec;
                prev_mean_depth = curr_mean_depth;

                curr_point.clear();
                curr_vec.clear();

                cv::imshow("Tracking", draw_mat);
                qDebug() << "The slow operation took" << timer.elapsed() << "milliseconds";
            }

            // Eliminate False Alarm every `FA_frame = 50` frames
            if(frame > 0 && frame % FA_frame == 0){
                int threshold = 2;
                QList<int> false_alarm = Eliminate_false_alarm(total_fruit, threshold, frame);
                qDebug() << frame << " eliminate: ";
                for(int i = 0 ; i < false_alarm.size() ; i++){
                    qDebug() << false_alarm[i];
                }
            }

            frame++;
            cv::waitKey(10);

            const uint64_t current_position = pipeline_profile.get_device().as<rs2::playback>().get_position();
            if( static_cast<int64_t>( current_position - last_position ) < 0 ){
                break;
            }
            last_position = current_position;
        }
    }

    int online_result = total_fruit.size();
    qDebug() << "After online" << online_result;

    // Eliminate false alarm
    int false_alarm_threshold = 2;
    QList<int> erase_ID = Eliminate_false_alarm(total_fruit, false_alarm_threshold);
    qDebug() << "After eliminate false alarm" << total_fruit.size();

    QString track_result_path = "./depth_data/tracking_result.csv";
    save_track_result(track_result_path, online_result, erase_ID, bag_filename);


    // Calculate Fruit Size Histogram
    qDebug() << "Calculating Fruit Size Histogram";
    std::pair<QList<int>, std::vector<std::pair<int, int>>> Histogram_max_min = Fruit_size_histogram(total_fruit);
    QList<int> Histogram = Histogram_max_min.first;
    std::vector<std::pair<int, int>> max_min = Histogram_max_min.second;

    QString save_path = "./depth_data/fruit_histogram.csv";
    save_histogram(save_path, Histogram, max_min);

    int max = max_min.at(0).second;
    int min = max_min.at(1).second;
    int bin_size = (max - min) / 5;
    int size_bin[6] = {min, min + bin_size, min + 2*bin_size, min + 3*bin_size, min + 4*bin_size, max};


    // Calculate Fruit Ripening Stage
    qDebug() << "Calculating Fruit Ripening Stage";
    QList<double> Stage_list = Fruit_ripening_stage(total_fruit);
    QString ripening_stage_path = "./depth_data/fruit_ripening_stage_histogram.csv";
    save_ripening_stage(ripening_stage_path, Stage_list);

    // Calculate global coordinate
    qDebug() << "Drawing Global map";
    cv::Point2f max_global(0.0, 0.0), min_global(10.0, 10.0);
    QList<global_coor> global_coord = Calculate_global_coordinate(total_fruit, max_global, min_global, Homo_history, true);

    if(min_global.x < 0)   max_global.x += abs(min_global.x);
    if(min_global.y < 0)   max_global.y += abs(min_global.y);
    cv::Mat Global_map((int)max_global.y + 1, (int)max_global.x + 1, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat Global_map_ellipse((int)max_global.y + 1, (int)max_global.x + 1, CV_8UC3, cv::Scalar(255, 255, 255));
    qDebug() << "Global_map.rows, cols: " << Global_map.rows << " " << Global_map.cols;

    // Find fruit with highest confidence score
//    float score = 0;
//    int n = 0;
//    std::pair<float, float> p_s_ratio;
//    for(int i = 0 ; i < total_fruit.size() ; i++){
//        float temp = total_fruit.at(i).prob;
//        qDebug() << total_fruit.at(i).track_id << "score: " << temp;
//        if(temp > score){
//            p_s_ratio = pixel_size_ratio(total_fruit.at(i));     // Width, Height: Pixel / Size
//            qDebug() << "p_s_ratio w, h" << p_s_ratio.first << p_s_ratio.second;
//            qDebug() << "W/H, H/W" << p_s_ratio.first / p_s_ratio.second << p_s_ratio.second / p_s_ratio.first;
//            if(tomato){
//                if(p_s_ratio.first / p_s_ratio.second < 2 && p_s_ratio.second / p_s_ratio.first > 1/2){     // Tomato: 2 / (1/2)
//                    score = temp;
//                    n = i;
//                }
//            }
//            else{
//                if(p_s_ratio.second > 2.5 * p_s_ratio.first){
//                    score = temp;
//                    n = i;
//                }
//            }
//        }
//    }
//    qDebug() << "HIGHEST confidence score ID, score: " << total_fruit.at(n).track_id << score;
//    qDebug() << "p_s_ratio w, h" << p_s_ratio.first << p_s_ratio.second;

    QFile actualsize("./depth_data/fruit_actual_size.csv");
    QFile actualsize_histo("./depth_data/fruit_actual_size_histogram.csv");
    QFile actualsize_wh("./depth_data/fruit_actual_size_wh.csv");
    QTextStream out(&actualsize);
    QTextStream out1(&actualsize_histo);
    QTextStream out2(&actualsize_wh);
    actualsize.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text);
    actualsize_histo.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text);
    actualsize_wh.open(QFile::WriteOnly|QIODevice::Append|QIODevice::Text);
    out << "======= Fruit actual size ======= \n";
    for(int i = 0 ; i < global_coord.size() ; i++){
        float global_x = global_coord.at(i).global_point.x;
        float global_y = global_coord.at(i).global_point.y;
        if(min_global.x < 0)   global_x += abs(min_global.x);
        if(min_global.y < 0)   global_y += abs(min_global.y);

        cv::Scalar color = set_maturity_color(global_coord.at(i).maturity);
        int radius = set_radius(global_coord.at(i).size, size_bin);

        cv::Point2d w_h = total_fruit.at(i).width_height.at(global_coord.at(i).nearest_trajectory_index);

        double actual_w = w_h.x * 0.003 * total_fruit.at(i).median_depth / 1.93;    // 0.003:  D435 pixel size(m), 1.93: D435 Focal Length
        double actual_h = w_h.y * 0.003 * total_fruit.at(i).median_depth / 1.93;    // (m)
        qDebug() << "nearest_index: " << global_coord.at(i).nearest_trajectory_index;
        qDebug() << "width pixel & actual: " << w_h.x << actual_w;
        qDebug() << "height pixel & actual: " << w_h.y << actual_h;

        out1 << 3.14159 * actual_w * actual_h / 4 << ",";
        out2 << actual_w << "," << actual_h << "\n";

        out << "ID: " << total_fruit.at(i).track_id << "\n";
        out << "   nearest_index: " << global_coord.at(i).nearest_trajectory_index << "\n";
        out << "   width pixel & actual: " << w_h.x << ", " << actual_w << "\n";
        out << "   height pixel & actual: " << w_h.y << ", " << actual_h << "\n";

        int x_axes = 200 * actual_w;
        int y_axes = 200 * actual_h;

        qDebug() << "x, y axes" << x_axes << y_axes;
        cv::ellipse(Global_map_ellipse, cv::Point(global_x, global_y), cv::Size(x_axes, y_axes), 0, 0, 360, color, -3);
        cv::putText(Global_map_ellipse, "ID: " + std::to_string(global_coord.at(i).global_fruit_ID), cv::Point(global_x - 80, global_y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 0, 255), 1);
        cv::circle(Global_map, cv::Point(global_x, global_y), radius, color, -1);
        cv::putText(Global_map, "ID: " + std::to_string(global_coord.at(i).global_fruit_ID), cv::Point(global_x - 80, global_y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 0, 255), 1);
    }
    actualsize.close();
    cv::imshow("Global map", Global_map);
    cv::imshow("Global map Ellipse", Global_map_ellipse);
    cv::imwrite("./depth_data/Global_map.jpg", Global_map);
    cv::imwrite("./depth_data/Global_map_ellipse.jpg", Global_map_ellipse);
    qDebug() << "End of Tracking";
}



inline void detect_with_depth::draw_color(){
    if(!color_frame)    return;

    const rs2_format color_format = color_frame.get_profile().format();
    switch( color_format ){
    case rs2_format::RS2_FORMAT_RGB8:{
        color_mat = cv::Mat( color_height, color_width, CV_8UC3, const_cast<void*>( color_frame.get_data() ) ).clone();
        cv::cvtColor( color_mat, color_mat, cv::COLOR_RGB2BGR );
        break;
    }
    case rs2_format::RS2_FORMAT_RGBA8:{
        color_mat = cv::Mat( color_height, color_width, CV_8UC4, const_cast<void*>( color_frame.get_data() ) ).clone();
        cv::cvtColor( color_mat, color_mat, cv::COLOR_RGBA2BGRA );
        break;
    }
    case rs2_format::RS2_FORMAT_BGR8:{
        color_mat = cv::Mat( color_height, color_width, CV_8UC3, const_cast<void*>( color_frame.get_data() ) ).clone();
        break;
    }
    case rs2_format::RS2_FORMAT_BGRA8:{
        color_mat = cv::Mat( color_height, color_width, CV_8UC4, const_cast<void*>( color_frame.get_data() ) ).clone();
        break;
    }
    case rs2_format::RS2_FORMAT_Y16:{
        color_mat = cv::Mat( color_height, color_width, CV_16UC1, const_cast<void*>( color_frame.get_data() ) ).clone();
        constexpr double scaling = static_cast<double>( std::numeric_limits<uint8_t>::max() ) / static_cast<double>( std::numeric_limits<uint16_t>::max() );
        color_mat.convertTo( color_mat, CV_8U, scaling );
        break;
    }
    case rs2_format::RS2_FORMAT_YUYV:{
        color_mat = cv::Mat( color_height, color_width, CV_8UC2, const_cast<void*>( color_frame.get_data() ) ).clone();
        cv::cvtColor( color_mat, color_mat, cv::COLOR_YUV2BGR_YUYV );
        break;
    }
    default:
        throw std::runtime_error( "unknown color format" );
        break;
    }
}

inline void detect_with_depth::draw_depth(){
    if( !depth_frame ){
        return;
    }
    depth_mat = cv::Mat( depth_height, depth_width, CV_16UC1, const_cast<void*>( depth_frame.get_data() ) ).clone();
}

inline void detect_with_depth::showColor()
{
    if(!color_frame)    return;
    if(color_mat.empty())   return;
    cv::imshow("Color", color_mat);
}

inline void detect_with_depth::showDepth(){
    if(!depth_frame)  return;
    if(depth_mat.empty())   return;

    cv::Mat scale_mat;
    cv::convertScaleAbs(depth_mat, scale_mat, 0.5);
    applyColorMap(scale_mat, scale_mat, cv::COLORMAP_JET);
    cv::imshow("Depth",scale_mat);
}

   void feature_processing_thread(cv::Mat prev_mat, cv::Mat curr_mat, std::vector<cv::Point2f>& prev_point, std::vector<cv::Point2f>& curr_point){
    {
        std::lock_guard<std::mutex> lock(point_mutex);
        std::vector<uchar> status;
        qDebug() << "threading:";

        std::vector<cv::Point2f>& pprev_point = prev_point;
        std::vector<cv::Point2f>& ccurr_point = prev_point;
        featureTracking(prev_mat, curr_mat, pprev_point, ccurr_point, status);

        //    qDebug() << "frame: " << frame;
        qDebug() << "prev_point, curr_point.size" << prev_point.size() << ", " << curr_point.size();

        if(prev_point.size() < 2000){
            featureDetection(prev_mat, pprev_point, 0);
            featureTracking(prev_mat, curr_mat, pprev_point, ccurr_point, status);
        }
        prev_point = pprev_point;
        curr_point = ccurr_point;
    }


   rs2::frame_queue postprocessed_frames;
   std::atomic_bool alive{true};

   std::thread video_processing_thread([&]() {
       rs2::processing_block frame_processor([&](rs2::frameset frameset, rs2::frame_source& source){
           frameset = align_to.process(frameset);
           rs2::frame depth = frameset.get_depth_frame();
           depth =

           source.frame_ready(frameset);
       });
       frame_processor >> postprocessed_frames;
       while(alive){
           rs2::frameset fs;
           if(pipeline.poll_for_frames(&fs))   frame_processor.invoke(fs);
       }
   });

   std::thread feature_thread(feature_processing_thread, prev_mat, curr_mat, std::ref(prev_point), std::ref(curr_point));

   while(true){
       static rs2::frameset current_frameset;
       postprocessed_frames.poll_for_frame(&current_frameset);
       if(current_frameset){
           auto color_framee = current_frameset.get_color_frame();
           uint32_t color_widthe = color_framee.as<rs2::video_frame>().get_width();
           uint32_t color_heighte = color_framee.as<rs2::video_frame>().get_height();
           color_mat = cv::Mat( color_heighte, color_widthe, CV_8UC3, const_cast<void*>( color_framee.get_data() ) ).clone();
           cv::cvtColor( color_mat, color_mat, cv::COLOR_RGB2BGR );

           auto depth_framee = frameset.get_depth_frame();
           //            uint32_t depth_widthe = depth_framee.as<rs2::video_frame>().get_width();
           //            uint32_t depth_heighte = depth_framee.as<rs2::video_frame>().get_height();
           //            depth_mat = cv::Mat( depth_widthe, depth_heighte, CV_16UC1, const_cast<void*>( depth_frame.get_data() ) ).clone();

           rs2::frame depth_frame_;
           depth_frame_ = depth_framee;
           cv::Mat draw_mat = color_mat.clone();
           cv::Mat maturity_mat = color_mat.clone();
           if((frame > 6) && (frame%1 == 0)){
               // Define current mat, point, vec
               curr_mat = color_mat.clone();
               std::vector<bbox_t> temp = detector.detect(curr_mat, detect_threshold);
               curr_vec = bbox_t2bbox_t_history(temp);
               temp.clear();



               std::vector<uchar> status;
               featureTracking(prev_mat, curr_mat, prev_point, curr_point, status);

               qDebug() << "frame: " << frame;
               qDebug() << "prev_point, curr_point.size" << prev_point.size() << ", " << curr_point.size();

               if(prev_point.size() < 2000){
                   featureDetection(prev_mat, prev_point, frame);
                   featureTracking(prev_mat, curr_mat, prev_point, curr_point, status);
               }



               // Current / Previous mat fruit
               QList<cv::Point2f> prev_fruit;
               draw_mat = draw_detect_point(curr_vec, draw_mat, fruit_point_color, draw_bbox);
               for(int i = 0 ; i < prev_vec.size() ; i++){
                   cv::Point2f temp((float)prev_vec[i].x + (float)prev_vec[i].w / 2, (float)prev_vec[i].y + (float)prev_vec[i].h / 2);
                   prev_fruit.append(temp);
               }

               // Kick fruit out of the boundary
               Kick_fruit_out_of_boundary(curr_vec, img_boundary, img_size, depth_frame, depth_threshold);

               cv::rectangle(draw_mat, cv::Point2f(img_boundary, img_boundary), cv::Point2f(draw_mat.cols - img_boundary, draw_mat.rows - img_boundary),
                             boundary_color, 1);

               qDebug() << "prev_point, curr_point.size" << prev_point.size() << ", " << curr_point.size();

               // Calculate and Draw tracked previous point
               cv::Mat homo_matrix = cv::findHomography(prev_point, curr_point);
               Homo_history.append(homo_matrix);
               for(int i = 0 ; i < prev_fruit.size() ; i++){
                   cv::Point2f temp_point = get_tracked_point(homo_matrix, prev_fruit.at(i));
                   cv::circle(draw_mat, temp_point, 5, tracked_point_color, -1);
                   cv::circle(draw_mat, temp_point, threshold, tracked_point_color, 1);
                   if(draw_bbox){
                       cv::rectangle(draw_mat, cv::Point2f((float)prev_vec[i].x, (float)prev_vec[i].y)
                                     , cv::Point2f((float)prev_vec[i].x + (float)prev_vec[i].w, (float)prev_vec[i].y + (float)prev_vec[i].h)
                                     , tracked_point_color, 1);
                   }
               }
               // Set ID with tracking result
               bool previous_fruit = true;
               if(prev_vec.size() == 0)    previous_fruit = false;
               bool saveIOU = false;
               set_ID_fast(total_fruit, prev_vec, curr_vec, Homo_history, previous_fruit, threshold, lost_track_threshold, draw_mat, maturity_mat, saveIOU);

               // Save history and Draw ID
               save_history(total_fruit, frame);
               draw_ID(curr_vec, draw_mat, total_fruit.size());
               cv::imwrite("./depth_data/" + std::to_string(frame) + ".jpg", draw_mat);

               prev_mat = curr_mat.clone();
               prev_point = curr_point;
               prev_vec = curr_vec;

               curr_point.clear();
               curr_vec.clear();

               cv::imshow("Tracking", draw_mat);
               cv::waitKey(33);
           }
           frame++;
           qDebug() << "main:";
       }
   }
   alive = false;
   video_processing_thread.join();
   feature_thread.join();
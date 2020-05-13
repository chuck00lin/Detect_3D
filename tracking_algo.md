# Fruit Tracking Algorithm 

`detecto_with_depth.cpp`

1. Environment
2. Architecture
	1. Flow 
	2. Data Structure
	3. Saved Data 
3. Function
    1. `detect_with_depth.cpp`
	2. `feature_function.hpp`
	3. `offline_tracking.hpp`
4. Others

## Environment
1. Qt 5.10.0 or Qt 5.7.1
2. MSVC 2015, 64 bits
3. Library
    1. OpenCV 3.2.0 with contribute and CUDA 8.0, build with MSVC 2015, 64 bits
    2. YOLO
        - Version 2.0
        - CUDA 8.0
        - cudnn-9.0-windows10-x64-v7.1
    3. Realsense SDK 2.16.5
    4. Opengl 3.2
    5. glfw3
4. Used dll
    1. CUDA/cudnn:
        ```
        cublas64_80.dll cufft64_80.dll curand64_80.dll 
        nppc64_80.dll nppi64_80.dll npps64_80.dll
        ```
    2. opengl/glfw: 
        ```
        glfw.dll opengl32sw.dll
        ```
    3. OpenCV with contribute/cuda:
        ```
        opencv_XXXXXX.dll opencv_cudaXXXXX.dll
        ```
    4. Realsense: `realsense2.dll`
    5. YOLO: `yolo_cpp_dll.dll`


## Architecture
### Flow
1. Read `.bag` file: `get_bag_file()`
2. Initialize Realsense: `initialize_realsense()`
3. Initialize Yolov2 Detector: `initialize_detector(bool tomato)`
	- tomato: True --> detect tomato
	- tomato: False --> detecto Strawberry
4. Initialize Parameters
--------------------------------- Online Tracking -------------------------------------------------
5. **Initialize Tracking**
	1. Read 2 frames for color and depth.
		- color: `frame0`, `frame1`
		- depth: `depth_frame0`, `depth_frame1`
	2. Detect fruits in color frame
		- `result_vec0`, `result_vec1` for `frame0`, `frame1`
	3. data structure transformation : from `bbox_t` to `bbox_t_history` (`detect_3d.hpp`)
	4. Erase unavailable fruits in detected result: `transform0`, `transform1`
	5. **Initialize `total_fruit` with `frame0` information**
	6. Feature Operation
		- `featureTracking_GPU` (`feature_function.hpp`)
		- Find camera movement: `cv::findHomography` or `cv::estimateRigidTransform`
		- Predict fruits position
	7. Set ID for each fruit: `set_ID_fast` (`feature_function.hpp`)
6. **Start Tracking**
	1. Frame preprocessing
	2. Repeat *5.Initialize Tracking*
	3. Eliminate false alarm every `FA_frame` frame: `Eliminate_false_alarm()` (`offline_tracking.hpp`)(Not necessary)
--------------------------------- Offline Tracking ------------------------------------------------
7. Eliminate false alarm: `Eliminate_false_alarm()` (`offline_tracking.hpp`)
8. Calculate fruit size histogram (pixel level): `Fruit_size_histogram()` (`offline_tracking.hpp`)
	- Fruit size (pixel level and true size) is calculated during tracking in function `Kick_fruit_out_of_boundary()`
9. Calculate fruit ripening stage: `Fruit_ripening_stage()` (`offline_tracking.hpp`)
10. Draw global map
	1. Calculate global coordinate: `Calculate_global_coordinate()` (`offline_tracking.hpp`)
	2. Draw map
		- Set ripening stage: `set_maturity_color()` (`offline_tracking.hpp`)
		- Set size (pixel level): `set_radius()` (`offline_tracking.hpp`)
		- Set size (true size): X (m) * 200
11. End

### Data Structure
1. bbox_t (defined by YOLO)
    ```cpp
    struct bbox_t{
        unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
        float prob;                 // confidence - probability that the object was found correctly
        unsigned int obj_id;        // class of object - from range [0, classes-1]
        unsigned int track_id;      // tracking id for video (0 - untracked, 1 - inf - tracked object)
    };
    ```

2. bbox_t_history (defined in detect_3d.hpp)
    ```cpp
    struct bbox_t_history:bbox_t{
        QList<unsigned int> history;                    // Fruit state history sequence: 0: Inactive, 1: Lost, 2: Tracked
        QList<cv::Point2f> trajectory;                  // Fruit center x, y coordinate when in Tracked state
        unsigned int lost_frame;                        // Frame number when the fruit is Lost
        QList<cv::Mat> frame_mat;                       // Save row frame for maturity calculation
        QList<cv::Point2d> width_height;                // Fruit width and height
        QList<double> depth_hist;                       // Fruit distance when in Tracked state
        QList<std::pair<float, float>> true_size_hist;  // Fruit true size calculated by 3D point cloud (NOT USED IN THE NEWEST VERSION)
        double maturity;                                // Fruit maturity. Calculated in offline stage
        int size;                                       // Fruit size in pixel value. Calculated in offline stage                       
        double median_depth;                            // Fruit depth calculate by the median value of all pixels in online stage runtime. Used for saving in QList<double> depth_hist.
        std::pair<float, float> true_size;              // Fruit true size calculated by 3D point cloud in online stage runtime. Used for saving in QList<std::pair<float, float>>  true_size_hist (NOT USED IN THE NEWEST VERSION)
    };
    ```

3. global_coor (defined in detect_3d.hpp)
    ```cpp
    struct global_coor{
        cv::Point2f global_point;           // Fruit global coordinate calculated in offline stage
        unsigned int global_fruit_ID;       // Fruit ID, same as bbox_t_history.track_ID
        double maturity;                    // Fruit maturity, same as bbox_t_history.maturity
        int size;                           // Fruit size, same as bbox_t_history.size
        int nearest_trajectory_index;       // Fruit trajectory index with smallest distance between the fruit and camera (NOT USED IN THE NEWEST VERSION)
    }
    ```

### Saved Data
1. fruit_actual_size.csv
    - for debugging
    - Ex. ======= Fruit actual size ======= 
          ID: 1
          nearest_index: 11
          width pixel & actual: 83, 0.0704425 (m)
          height pixel & actual: 95, 0.0806269 (m)

2. fruit_actual_size_histogram.csv
    - Thesis fig. 4-27(c), by GraphPad Prism 7
    - Ex. 0.0178429,0.0182098,0.0175331,0.0071291,0.00810463,0.0147567,0.0160196,

3. fruit_actual_size_wh.csv
    - Thesis fig. 4-27(a), by GraphPad Prism 7
    - Ex. 0.0704425,0.0806269
          0.079744,0.0726871
          0.0742896,0.0751244
          0.0480529,0.0472244
          0.0519684,0.0496415

4. fruit_histogram.csv
    - Pixel size
    - Thesis fig. 4-27(b), by GraphPad Prism 7
    - Ex. ======= Fruit size histogram ======== 
          9310, 11865, 8134, 3410, 4830, 9951, 9310, 10486, 11021, 8096, 4180, 
          Max ID, size: 2, 11865

          Max ID, size: 4, 3410

5. fruit_ripening_stage_histogram.csv
    - Thesis fig. 4-26(b), by GraphPad Prism 7
    - Ex. ======= Fruit ripening stage histogram ======= 
          1, 3, 4, 2, 4, 1, 2, 1, 4, 3, 3, 

6. history.csv
    - stage sequence for each fruit ID
    - for debugging
    - Ex. Frame : 4
          ID: 1  2 2 1 
          Frame : 5
          ID: 1  2 2 1 2 
          Frame : 6
          ID: 1  2 2 1 2 1 
          Frame : 7
          ID: 1  2 2 1 2 1 1 
          Frame : 8
          ID: 1  2 2 1 2 1 1 1 
          Frame : 9
          ID: 1  2 2 1 2 1 1 1 1 
          Frame : 10
          ID: 1  2 2 1 2 1 1 1 1 2 

7. maturity_HSV_histogram.csv
    - Hue pixel value for each fruit
    - Thesis fig. 4-26(a), by GraphPad Prism 7
    - Ex. 18,18,23,15,19,19,23,23,18,18,23,23,24,24,21,21,19,19,21,21,19,19,21,21,27,27,27,27,...

8. tracking_result.csv
    - for debugging and analyzing data
    - Ex. ======= Tracking Result =======
          Video: F:/Fruit_harvest/Raw_data/08152019_Bird/20190815_up06.bag
          Online-tracking
          total_fruit.size() = 17

          Offline-tracking
          Erase-ID
          7, 8, 10, 12, 16, 17, 
          total_fruit.size() = 11

9. Global_map.jpg
    - pixel size
    - Thesis fig. 4-28 Relative Size

10. Global_map_ellipse_ID_24.jpg
    - True size
    - Thesis fig. 4-28 Actual Size


## Function
### detect_with_depth.cpp
    1. get_bag_file()
    2. initialize_realsense()
    3. initialize_detector()
        - type=(type==True)?tomato:strawberry
        - model
            - 1 vs 1 model: trained by `20171226_JingYong` imgs
            - 1st round data: mostly trained by `01112019_GaoXiaw_Jinyong_depth`
            - 1st round data (modified): modified model by tuning the size and number of the anchor boxes
            - internet: data from `From_internet`
            - 2nd round data: data from `04112019_JinYong`
            - Only 100% model: fruit labeled only 100% visible. data from `01112019_GaoXiaw_Jinyong_depth`
    4. Color()
    5. Depth()
    6. draw_detect_point(std::vector<bbox_t> result_vector, cv::Mat input, cv::Scalar fruit_point_color, bool bbox)
        - bool bbox: draw bounding box boundary or not
    7. draw_detect_point(std::vector<bbox_t_history> result_vector, cv::Mat input, cv::Scalar fruit_point_color, bool bbox)
    8. mean_distance() (X)
        - mean distance of all *fruit* pixels
    9. fame_mean_distance()
        - mean distance of all *frame* pixels
    10. cal_SD()
        - standard deviation of distance of all *fruit* pixels
    11. median_distance()
        - meidan distance of all *fruit* pixels
        - `cal_SD()`
        - return std::make_pair(median distance, SD)
    12. drawHistImg() (X)
        - draw img HSV histogram
    13. dist_3d
        - calculate distance between 2 points by 3D point cloud
    14. fruit_true_size()
        - `dist_3d`
    15. **Kick_fruit_out_of_boundary()**
        - Elimiate detection results if out of boundary. boundary = 40
        - `median_distance()`
        - if(median_distance < lower depth threshold ||     // lower depth threshold = 0.3 (m)
             median_distance > higher depth threshold ||    // lower depth threshold = 0.9 (m)
             SD_distance > SD_threshold)                    // SD_threshold = 0.3, maybe leaves
            -> Elimiate detection results
    16. draw_ID()
    17. save_history()
    18. set_saveiou() (X)
    19. pixel_size_ratio(X)
    20. moving_degree(X)
    21. preprocess (X)
        - input frame preprocessing: output = input frame * alpha
    22. postprocess (X)
    23. **run()**
    24. draw_color()
    25. draw_depth()
    26. showColor()
    27. showDepth()

### feature_function.hpp
    1. featureDetection (X)
    2. featureDetection_GPU (X)
    3. featureTracking (X)
    4. featureTracking_GPU
    5. get_tracked_point
    6. global_coordinate
    7. set_ID
    8. IOU(bbox_t_history prev_vec, bbox_t_history curr_vec, cv::Mat homo)
    9. IOU(cv::Point2f prev_pt_LT, cv::Point2f curr_pt_LT, cv::Point2f prev_pt_RB, cv::Point2f curr_pt_RB)
        - LT: left top corner coordinate
        - RB: right bottom corner coordinate

    10. set_ID_fast(std::vector<bbox_t_history>& total_fruit
                    , std::vector<bbox_t_history>& prev_vec
                    , std::vector<bbox_t_history>& curr_vec
                    , QList<cv::Mat> Homo_history
                    , QList<double> mean_depth_diff
                    , bool prev_fruit
                    , QList<int> threshold
                    , int lost_track_threshold (X)
                    , QList<float> avg_point_dist_hist
                    , cv::Mat& check_mat
                    , cv::Mat maturity_mat
                    , bool save_IOU
                    , bool depth)

        - `total_fruit`
        - `prev_vec`: detected result of previous frame
        - `curr_vec`: detected result of current frame
        - `Homo_history`: rigid transform matrix series, Homo_history.size() = current frame
        - `mean_depth_diff`: mean depth difference between two frames series
        - `prev_fruit`: if there is fruit in the previous frame or not
        - `threshold`: stage 1, radius for each fruit in the previous frame
        - `lost_track_threshold`: (X)
        - `avg_point_dist_hist`: average distance of the feature points history
        - `check_mat`: a.k.a. draw_mat, used to draw results
        - `maturity_mat`: save row frame when 'Tracked'
        - `save_IOU`: save IOU results or not
        - `depth`: take depth information into account for decision
        ```cpp

        // -------- STAGE ONE --------- //
        for(int i = 0 ; i < curr_vec.size() ; i++){
        	// check the fruit is too closed with others or not.
        	//	| - Y:	decrease the threshold
        	//	| - N:  do nothing
        	if(prev_fruit){
        		for(int j = 0 ; j < prev_vec.size() ; j++){
        			// Set threshold    			
        			// Calculate distance: L2norm(prev_fruit, curr_fruit)
        			// Calculate IOU
        			// Calculate depth_diff: abs(prev_fruit.median_depth - curr_fruit.median_depth)
        			// Set ID decision:
        			//	| - Smallest distance
        			//	| - IOU > IOU_threshold
        			//	| - depth_diff < 0.05
        			if(success){
                        //	ith fruit in current frame equals to jth fruit in previous frame
        				cout << "First stage - tracked  ID:" << prev_vec.at(j).track_id;
        			}
                    else{
                        // ith fruit in current frame may be new a fruit or lost2track fruit
                        cout << "First stage - Lost2Track or New";
                    }
        		}
                if(Tracked){
                    // Save total_fruit information
                }
        	}
            else{   // Must be a new fruit
                // Save total fruit information
                curr_vec.at(i).track_id = total_fruit.size() + 1;
                cout << "First stage - New Fruit (No Fruit in last frame)  ID: " << curr_vec.at(i).track_id;
            }
        }

        // Append history with lost and mark the lost frame
        for(int i = 0 ; i < total_fruit.size() ; i++){
            total_fruit.at(i).lost_frame = curr_frame;
        }

        // -------- STAGE TWO --------- //
        for(int i = 0 ; i < curr_vec.size() ; i++){
            if("first stage - Lost2Track or New"){
                for(int j = 0 ; j < total_fruit.size() ; j++){
                    if(total_fruit.at(j) == LOST){
                        // Predict its position, bbox and depth in current frame by Homo_history and mean_depth_diff. Used Information:
                        // | - total_fruit.trajectory 
                        // | - total_fruit.width_height
                        // | - total_fruit.median_depth
                        // | - Homo_history
                        // | - mean_depth_diff

                        // Check the predicted position, if out of image boundary then set to 'Inactive State'
                    }
                    // Calculate distance: L2norm(prev_fruit, curr_fruit)
                    // Calculate IOU
                    // Calculate diff_depth: abs(prev_fruit.median_depth - curr_fruit.median_depth)
                }
                // Set ID decision:
                //  | - Smallest distance
                //  | - IOU > IOU_threshold
                //  | - depth_diff < 0.05
                if(there are lost fruits in total_fruit){
                    if(success){
                        //  ith fruit in current frame equals to the fruit in Lost state
                        cout << "Second stage - 1. Lost -> Tracked";
                        // Save total fruit information
                    }
                    else{
                        // ith fruit in current frame is truly a new fruit
                        cout << "Second stage - 2. New Fruit";
                        // Save total fruit information
                    } 
                }
                else{
                    // ith fruit in current frame is a new fruit
                    cout << "Second stage - 2. New Fruit (No Fruit Lost in total fruit)"
                    // Save total fruit information
                }
            }
        }
        ```

        * Save total fruit information
            * track_id
            * history: 1: Lost, 2: Tracked, 3: Inactive
            * trajectory: center point of bbox
            * frame_mat: row_frame, for calculating ripening stage
            * width_height: w, h, for calculating fruit ripening stage and fruit size
            * depth_hist: median_depth of fruit, for finding the nearest fruit frame
            * true_size_hist: true size history, for calculating fruit size
        * closeness_threshold: handling the circumstance if fruits are too closed
        * duplicate, used_ids: handling duplicated ID

	11. bbox_t2bbox_t_history()

### offline_tracking.hpp
    1. **save_track_result()**
    2. **save_histogram()**
    3. **save_ripening_stage()**
    4. max_fruit_frame_size()
    5. **Eliminate_false_alarm(std::vector<bbox_t_history>&, int threshold)**
    6. **Eliminate_false_alarm(std::vector<bbox_t_history>&, int threshold, int frame)**
    7. **Fruit_size_histogram()**
        - max_fruit_frame_size()
        - set `total_fruit.size`
    8. maturity()
        - Calculate maturity
    9. save_ripen_img()
        - Draw fruit mask with ripening stage
    10. ripening_stage()
        - Grabcut algorithm
        - maturity()
        - save_ripen_img()
    11. **Fruit_ripening_stage()**
        ```cpp
        for(each fruit){
            double stage = ripening_stage()
            total_fruit.at(i).maturity = stage;
        }
        ```
    12. set_coordinate()
        - global_coor
        ``` cpp
        global_coor coor;
        coor.global_point = point;
        coor.global_fruit_ID = total_fruit.track_id;
        coor.maturity = total_fruit.maturity;
        coor.size = total_fruit.size;
        ```
    13. near_origin_point()
    14. nearest_point()
    15. **Calculate_global_coordinate()**
        ```cpp
        for(each fruit){
            trajectory_index = nearest_point() or near_origin_point()
            global_coordinate() // defined in feature_function.hpp
            set_coordinate()
            coor.nearest_trajectory_index = trajectory_index
        }
        ```
    16. **set_maturity_color()**
    17. **set_radius()**


## Others
1. Fruit matching part can be advanced by Hungarian Algorithm
    [Hungarian Algorithm]: https://en.wikipedia.org/wiki/Hungarian_algorithm
2. Which frame to draw a fruit on global map? 
3. Fruit may be tilt. Bounding box width and height cannot truely describe the fruit size.
4. Map Pixel length vs Actual length miss match in Global map, result from rigid transform matrix.
5. Time complexity bottle neck
    - main: feature tracking (using optical flow) -> traditional tracking algo. !?
    - second: calculating the predicted position of 'Lost' fruit -> can be advanced with hungarian algo.
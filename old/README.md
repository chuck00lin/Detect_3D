# Fruit-Tracking-with-Realsense-D435

## Tracking fruits with realsense D435

### Function
1. Get distance: get the distance of objects detected by yolo detector
    - ybias means the result text shown below the center of the object
2. Draw Point cloud: use GLFW to draw point cloud
3. Epipolar Geometry: draw epipolar line between two image using cv::findFundamentalMat
4. Camera Pose:
    - Refer to [Avi Singh's blog] (https://avisingh599.github.io/vision/monocular-vo/)
    - Drawing visual odometry
    - Scale: space between continuous point in 3D
    - Frame: fps to calculate once
5. Tracking
    - Draw Tracking Line: Just for testing
    - Tracking: with bug
    - Faster Tracking: when the tracked point < 2000, re-detect the image feature points

### .dll must include:
#### 1. CUDA: 
    cublas64_80.dll curand64_80.dll
#### 2. GLFW: 
    glfw3.dll opengl32sw.dll glfw3dll.exp (optional) 
#### 3. openCV:
    opencv_core320.dll opencv_core320d.dll opencv_cudaarithm320.dll opencv_cudaarithm320d.dll
    opencv_cudev320.dll opencvcudev320d.dll opencv_features2d320.dll opencv_features2d320d.dll
    opencv_ffmpeg320_64.dll opencv_flann320.dll opencv_flann320d.dll opencv_imgproc320.dll  
    opencv_improc320d.dll
    opencv_world320.dll opencv_world320d.dll opencv_xfeatures2d320.dll opencv_xfeatures2d320d.dll
#### 4. YOLOv2: 
    yolo_cpp_dll.dll
#### 5. Realsense: 
    realsense2.dll
#### 6. Qt core: 
    D3Dcompiler_47.dll libEGL.dll libGLESV2.dll Qt5Core.dll Qt5Gui.dll Qt5Svg.dll Qt5Widgets.dll

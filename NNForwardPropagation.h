#pragma once
#include "cv.h"
#include <ctime>   
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"


using namespace std;
using namespace cv;
using namespace cv::dnn;


Mat getPlane(const Mat &m, int n, int cn);
void imgFromBlob(const Mat& blob_, OutputArrayOfArrays images_);
bool LoadOriginImg(string m_filepath, Mat& dst);
void showdicom(Mat I);
void FaceDetector_opencv_caffe();
void testCNN(string dcm_filepath);
//#pragma once
#ifndef __NNFORWARDPROPAGATION__
#define __NNFORWARDPROPAGATION__
#include "cv.h"
#include <ctime>   
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"


using namespace std;
//using namespace cv;
//using namespace cv::dnn;


cv::Mat getPlane(const cv::Mat &m, int n, int cn);
void imgFromBlob(const cv::Mat& blob_, cv::OutputArrayOfArrays images_);
bool LoadOriginImg(string m_filepath, cv::Mat& dst);
bool LoadOriginImg(string m_filepath, cv::Mat& dst, uint RowStartIdx, uint RowEndIdx, uint ColStartIdx, uint ColEndIdx);
void showdicom(cv::Mat I);
void FaceDetector_opencv_caffe();
void testCNN(string dcm_filepath);
void testCNN_TF(string dcm_filepath);
void Denoise_CNN(int rows, int cols, int nimg, int type, short* data);
bool SaveMatFile(cv::Mat I);

#endif
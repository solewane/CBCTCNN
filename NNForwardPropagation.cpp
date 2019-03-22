#include "stdafx.h"
#include "NNForwardPropagation.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;


void FaceDetector_opencv_caffe() {



	// Set the size of image and meanval
	const size_t inWidth = 300;
	const size_t inHeight = 300;
	const double inScaleFactor = 1.0;
	const Scalar meanVal(104.0, 177.0, 123.0);




	// Load image
	Mat img;
	img = imread("D:/360极速浏览器下载/微信图片_20190315162827.png");



	// Initialize Caffe network
	float min_confidence = 0.5;

	String modelConfiguration = "D:/360极速浏览器下载/face_detector/face_detector/deploy.prototxt";

	String modelBinary = "D:/360极速浏览器下载/face_detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel";

	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);



	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt:   " << modelConfiguration << endl;
		cerr << "caffemodel: " << modelBinary << endl;
		exit(-1);
	}
	Mat inputBlob = blobFromImage(img, inScaleFactor, Size(inWidth, inHeight), meanVal, false, false);
	net.setInput(inputBlob, "data");    // set the network input
	Mat detection = net.forward("detection_out");    // compute output

													 // Calculate and display time and frame rate

	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;  //time counting

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	ostringstream ss;
	ss << "FPS: " << 1000 / time << " ; time: " << time << "ms" << endl;

	putText(img, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

	float confidenceThreshold = min_confidence;
	for (int i = 0; i < detectionMat.rows; ++i)
	{
		// judge confidence
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);
			Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));
			//Rect object((int)xLeftBottom, (int)yLeftBottom, (int(xRightTop - xLeftBottom),
			// (int)(yRightTop - yLeftBottom));
			rectangle(img, object, Scalar(0, 255, 0));
			ss.str("");
			ss << confidence;
			String conf(ss.str());
			String label = "Face: " + conf;
			int baseLine = 0;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			rectangle(img, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
				Size(labelSize.width, labelSize.height + baseLine)),
				Scalar(255, 255, 255), CV_FILLED);
			putText(img, label, Point(xLeftBottom, yLeftBottom),
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		}
	}

	namedWindow("Face Detection", WINDOW_NORMAL);
	imshow("Face Detection", img);
	waitKey(0);
}

void testOpenCV() {
	IplImage* test;
	test = cvLoadImage("D:/360极速浏览器下载/微信图片_20190315162827.png", 1);
	cvNamedWindow("opencv_demo", 1);
	cvShowImage("opencv_demo", test);
	cvWaitKey(0);
	cvDestroyWindow("opencv_demo");
}

bool LoadOriginImg(string m_filepath, Mat& dst)
{
	if (m_filepath.length() == 0)
	{
		return false;
	}

	DicomImage *img = new DicomImage(m_filepath.c_str());

	Uint16 *pixel = (Uint16*)(img->getOutputData(16));

	cv::Mat dst2(img->getWidth(), img->getHeight(), CV_16U, pixel);

	Range R1;
	R1.start = 0;
	R1.end = 300; //300x300 size

	Mat mask = Mat::Mat(dst2, R1, R1);
	dst = mask;

	return true;
}


Mat getPlane(const Mat &m, int n, int cn)
{
	CV_Assert(m.dims > 2);
	int sz[32];
	for (int i = 2; i < m.dims; i++)
	{
		
		sz[i - 2] = m.size.p[i];
	}
	
	return Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));
}

void imgFromBlob(const Mat& blob_, OutputArrayOfArrays images_)
{
	//blob 是浮点精度的4维矩阵
	//blob_[0] = 批量大小 = 图像数
	//blob_[1] = 通道数
	//blob_[2] = 高度
	//blob_[3] = 宽度    
	CV_Assert(blob_.depth() == CV_32F);
	CV_Assert(blob_.dims == 4);

	//images_.create(cv::Size(1, blob_.size[0]),blob_.depth() );//多图，不明白为什么？
	images_.create(blob_.size[2], blob_.size[3], blob_.depth());//创建一个图像
	std::vector<Mat> vectorOfChannels(blob_.size[1]);
	//for (int n = 0; n <  blob_.size[0]; ++n) //多个图
	{int n = 0;                                //只有一个图
		for (int c = 0; c < blob_.size[1]; ++c)
		{
			
			vectorOfChannels[c] = getPlane(blob_, n, c);
		}
		//cv::merge(vectorOfChannels, images_.getMatRef(n));//这里会出错，是前面的create的原因？
		cv::merge(vectorOfChannels, images_);//通道合并
	}
}

void showdicom(Mat I)
{
	double maxx = 0, minn = 0;
	double *max = &maxx;
	double *min = &minn;
	I.convertTo(I, CV_64FC1);
	minMaxIdx(I, min, max);
	for (int i = 0; i<I.rows; i++)
	{
		for (int j = 0; j<I.cols; j++)
		{
			I.at<double>(i, j) = 255 * (I.at<double>(i, j) - minn) * 1 / (maxx - minn);
		}
	}

	imshow("DICOM Image", I);
	waitKey(0);
}


void testCNN(string dcm_filepath) {

	// Set the size of image and meanval
	const size_t inWidth = 300;
	const size_t inHeight = 300;
	const double inScaleFactor = 1.0 / 3000;
	const Scalar meanVal(0, 0, 0);

	String modelConfiguration = "D:/zyx/降噪/RED-CNN/Red_CNN.prototxt";

	String modelBinary = "D:/zyx/降噪/RED-CNN/Red_CNN.caffemodel";

	clock_t start = clock();
	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);



	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt:   " << modelConfiguration << endl;
		cerr << "caffemodel: " << modelBinary << endl;
	}
	printf_s(" model path = %s\nconfig path = %s\n", modelConfiguration.c_str(), modelBinary.c_str());

	Mat img;
	//bool isLoad = LoadOriginImg("D:/zyx/降噪/RED-CNN/0206.dcm", img);
	bool isLoad = LoadOriginImg(dcm_filepath, img); 
	printf_s("dicom loading path = %s\n", dcm_filepath.c_str());
	if (!isLoad) cout << "Dicom loading failed" << endl;
	img.convertTo(img, CV_32F); //convert from CV_16U to CV_32F  
	//cout << img2 - 31768 << endl;





	// Prepare blob
	//Mat img2 = img - 32768 + 1000;
	Mat inputBlob = blobFromImage(img - 32768 + 1000, inScaleFactor, Size(inWidth, inHeight), 0, false, false, CV_32F);
	//Mat inputBlob = blobFromImage(img);
	net.setInput(inputBlob, "data");    // set the network input
										// compute output




	Mat outputBlob = net.forward("etRelu2");//"etRelu2"				 
											//net.forward();




	//**time counting
	//vector<double> layersTimings;
	//double freq = getTickFrequency() / 1000;
	//double time = net.getPerfProfile(layersTimings) / freq;
	//printf_s("dicom loading path = %f\n", time/1000);
	Mat Outimg;
	//imagesFromBlob(outputBlob, Outimg);

	//cvNamedWindow("opencv_demo", 1);
	//Mat outputImg = (double)*outputBlob.data;
	//imshow("Face Detection", outputImg);	


	imgFromBlob(outputBlob, Outimg);

	clock_t duration_Output = clock();

	cout << "elapsed time："
		<< (double)(duration_Output - start) / CLOCKS_PER_SEC << "秒" << endl;


	//cout << Outimg-(img2 - 31768)/3000 << endl;


	//namedWindow("Out", WINDOW_NORMAL);
	//showdicom(img2);
	//namedWindow("In", WINDOW_NORMAL);
	imshow("In", (img - 31768) / 3000);
	imshow("Out", Outimg);
	imshow("diff", (img - 31768) / 3000 - Outimg);
	//showdicom(Outimg);

	waitKey(0);
	//img.release();  //mat不需要手动释放


}

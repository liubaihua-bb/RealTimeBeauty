#include "define.h"
#include "VideoFaceDetector.h"

int main(int argc, char** argv)
{
	// 打开默认摄像头
	VideoCapture camera;
	camera.open(0);
	//camera.open("D:\\video.mp4");
	if (!camera.isOpened()) {
		cout << "camera not open." << endl;
		return 1;
	}

	namedWindow("Camera video", WINDOW_KEEPRATIO | WINDOW_AUTOSIZE);

	VideoFaceDetector detector(camera);
	Mat frame;
	double fps = 0, time_per_frame;
	while (true)
	{
		auto start = cv::getCPUTickCount();
		detector >> frame;
		detector.beauty((void*)(&frame));

		auto end = cv::getCPUTickCount();//测量运行时间

		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);

		imshow("Camera video", frame);
		if (waitKey(25) == 27) break;
	}
	return 0;
}
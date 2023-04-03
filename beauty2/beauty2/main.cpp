#include <opencv2/opencv.hpp>
#include "demo.h"
#include <iostream>


using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("E:/face/face2.jfif"); //  B, G, R
	if (src.empty()) {
		printf("could not load image....\n");
		return -1;
	}
	// namedWindow("输入窗口", WINDOW_FREERATIO);
	imshow("输入窗口", src);

	QuickDemo qd;
	qd.bitwise_demo(src);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
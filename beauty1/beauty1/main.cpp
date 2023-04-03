#include "define.h"
#include "beauty.h"


int main(int argc, char** argv)
{
	/*string path = "E:/face/face2.jfif";
	E:/face/face1.jpeg
	Mat src = imread(path); //  B, G, R
	if (src.empty()) {
		printf("could not load image....\n");
		return -1;
	}*/
	//namedWindow("输入窗口", WINDOW_FREERATIO);
	//imshow("输入窗口", src);

	Beauty test1;
	/*Mat frame;
	VideoCapture capture;
	//读取视摄像头实时画面数据，0默认是笔记本的摄像头
	capture.open(0);   //打开摄像头 
	//capture.open("1.mp4");   //打开视频  
	if (!capture.isOpened())
	{
		std::cout << "video not open." << std::endl;
		return 1;
	}

	while (true)
	{

		capture >> frame;            //读取当前帧
		if (!frame.empty()) {          //判断输入的视频帧是否为空的

			imshow("window", frame);  //在window窗口显示frame摄像头数据画面
		}


		if (waitKey(20) == 'q')   //延时20ms,获取用户是否按键的情况，如果按下q，会退出程序 
			break;
	}

	capture.release();     //释放摄像头资源
	destroyAllWindows();   //释放全部窗口*/

	waitKey(0);
	cv::destroyAllWindows();
	return 0;
}









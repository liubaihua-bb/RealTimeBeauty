
#include <opencv2\opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;


int main()
{
	Mat frame;
	VideoCapture capture;
	//读取视摄像头实时画面数据，0默认是笔记本的摄像头
	capture.open(0);   //打开摄像头 
	//capture.open("1.mp4");   //打开视频  
	if (!capture.isOpened())
	{
		std::cout << "video not open." << std::endl;
		return 1;
	}

	/*	double rate = capture.get(CAP_PROP_FPS);     	//获取当前视频帧率
	//当前视频帧
	Mat curframe;

	//每一帧之间的延时
	int delay = 1000 / rate;      	//与视频的帧率相对应
	bool stop(false);
	while (!stop)
	{
		if (!capture.read(frame))   //获取视频或摄像头的每一帧
		{
			std::cout << "no video frame" << std::endl;
			break;
		}

		//此处为添加对视频的每一帧的操作方法
		int frame_num = capture.get(CAP_PROP_POS_FRAMES);
		std::cout << "Frame Num : " << frame_num << std::endl;
		if (frame_num == 500)                     //获取500帧
		{
			capture.set(CAP_PROP_POS_FRAMES, 10);      //重新设置帧数  重头播放
		}

		cv::imshow("video", frame);
		//引入延时
		//也可通过按键停止
		if (cv::waitKey(delay) > 0)
			stop = true;
	}
	//关闭视频，手动调用析构函数（非必须）
	capture.release();
	return 0;
*/


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
	destroyAllWindows();   //释放全部窗口
	return 0;
}
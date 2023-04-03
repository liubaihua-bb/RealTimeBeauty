
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
	//��ȡ������ͷʵʱ�������ݣ�0Ĭ���ǱʼǱ�������ͷ
	capture.open(0);   //������ͷ 
	//capture.open("1.mp4");   //����Ƶ  
	if (!capture.isOpened())
	{
		std::cout << "video not open." << std::endl;
		return 1;
	}

	/*	double rate = capture.get(CAP_PROP_FPS);     	//��ȡ��ǰ��Ƶ֡��
	//��ǰ��Ƶ֡
	Mat curframe;

	//ÿһ֮֡�����ʱ
	int delay = 1000 / rate;      	//����Ƶ��֡�����Ӧ
	bool stop(false);
	while (!stop)
	{
		if (!capture.read(frame))   //��ȡ��Ƶ������ͷ��ÿһ֡
		{
			std::cout << "no video frame" << std::endl;
			break;
		}

		//�˴�Ϊ��Ӷ���Ƶ��ÿһ֡�Ĳ�������
		int frame_num = capture.get(CAP_PROP_POS_FRAMES);
		std::cout << "Frame Num : " << frame_num << std::endl;
		if (frame_num == 500)                     //��ȡ500֡
		{
			capture.set(CAP_PROP_POS_FRAMES, 10);      //��������֡��  ��ͷ����
		}

		cv::imshow("video", frame);
		//������ʱ
		//Ҳ��ͨ������ֹͣ
		if (cv::waitKey(delay) > 0)
			stop = true;
	}
	//�ر���Ƶ���ֶ����������������Ǳ��룩
	capture.release();
	return 0;
*/


	while (true)
	{

		capture >> frame;            //��ȡ��ǰ֡
		if (!frame.empty()) {          //�ж��������Ƶ֡�Ƿ�Ϊ�յ�

			imshow("window", frame);  //��window������ʾframe����ͷ���ݻ���
		}


		if (waitKey(20) == 'q')   //��ʱ20ms,��ȡ�û��Ƿ񰴼���������������q�����˳����� 
			break;
	}

	capture.release();     //�ͷ�����ͷ��Դ
	destroyAllWindows();   //�ͷ�ȫ������
	return 0;
}
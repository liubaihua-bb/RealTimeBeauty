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
	//namedWindow("���봰��", WINDOW_FREERATIO);
	//imshow("���봰��", src);

	Beauty test1;
	/*Mat frame;
	VideoCapture capture;
	//��ȡ������ͷʵʱ�������ݣ�0Ĭ���ǱʼǱ�������ͷ
	capture.open(0);   //������ͷ 
	//capture.open("1.mp4");   //����Ƶ  
	if (!capture.isOpened())
	{
		std::cout << "video not open." << std::endl;
		return 1;
	}

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
	destroyAllWindows();   //�ͷ�ȫ������*/

	waitKey(0);
	cv::destroyAllWindows();
	return 0;
}









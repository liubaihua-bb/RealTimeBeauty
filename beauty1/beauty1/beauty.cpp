
#include "beauty.h"


Beauty* Beauty::m_pIntance = nullptr;//�����ⶨ�壬��cpp�ж���

Beauty::Beauty()
{
	//��ʼ����̬����
	m_pIntance = this;//��ָ��ָ��ö��󣡣���
	//��ʼ��������
	initMainImgUI();
}
 

void Beauty::initMainImgUI()
{
	string path;
	cout << "������ͼƬ�ľ���·����" << endl;
	cin >> path;

	namedWindow("Beauty", WINDOW_AUTOSIZE);
	
	m_MainImg = imread(path);
	imshow("src", m_MainImg);//ԭͼ

	//�����������68��
	m_vecFaceData = dectectFace68(path);

	int lignhtness = 100;
	int contrast = 100;
	int bigeye = 50;
	int thinface = 50;
	int beautyface = 100;
	//createTrackbar("����", "Beauty", &lignhtness, 100, on_lightness, (void*)(&m_MainImg));
	//createTrackbar("�Աȶ�", "Beauty", &contrast, 200, on_contrast, (void*)(&m_MainImg));
	//createTrackbar("����", "Beauty", &bigeye, 100, on_BigEye, (void*)(&m_MainImg));
	createTrackbar("����", "Beauty", &thinface, 100, on_thinFace, (void*)(&m_MainImg));
	//createTrackbar("ĥƤ", "Beauty", &beautyface, 200, on_beautyFace, (void*)(&m_MainImg));

	on_lightness(100, (void*)(&m_MainImg));
	//imshow("Beauty", m_MainImg);
	//waitKey(0);

}

//�ص����� ���ȵ��ڵ�ʵ��
void Beauty::on_lightness(int b, void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	//m = Scalar(b, b, b);
	addWeighted(img, 1.0, m, 0, b, dst);
	imshow("Beauty", dst);
	waitKey(0);
}

//�ص����� �Աȶȵ��ڵ�ʵ��
void Beauty::on_contrast(int b, void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	//m = Scalar(b, b, b);
	double con = b / 100.0;
	addWeighted(img, con, m, 0, 0, dst);//addWeighted()�����ǽ�������ͬ��С,��ͬ���͵�ͼƬ�ںϵĺ���
	imshow("Beauty", dst);
}




//�������ݼ�� ͨ������������dlib����ȡ68�����Ե����ݵ�
std::vector<std::vector<Point2f>> Beauty::dectectFace68(const string &path)
{
	std::vector<std::vector<Point2f>>  rets;
	//����ͼƬ·��
	array2d<rgb_pixel> img;
	cv_image<bgr_pixel> cimg(m_MainImg);
	//load_image(img, path.c_str());
	//�������������
	frontal_face_detector detector = get_frontal_face_detector();
	std::vector<dlib::rectangle> dets = detector(cimg);

	for (auto var : dets)
	{
		//�ؼ�������
		shape_predictor sp;
		deserialize("D:\\dlib19.22\\dlib-19.22\\shape_predictor_68_face_landmarks.dat") >> sp;
		//����shape���󱣴����68���ؼ���
		full_object_detection shape = sp(cimg, var);
		//�洢�ļ�
		ofstream out("face_detector.txt");
		//��ȡ�ؼ��㵽������
		std::vector<Point2f> points_vec;
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			auto a = shape.part(i);
			out << a.x() << " " << a.y() << " ";
			Point2f ff(a.x(), a.y());
			points_vec.push_back(ff);
		}
		rets.push_back(points_vec);
	}
	cout << "����������:" << dets.size() << "����������" << endl;
	return rets;
}

//�ص����� ����Ч������

void Beauty::on_BigEye(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat dst = src.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f left_landmark = points_vec[38];
		Point2f	left_landmark_down = points_vec[27];

		Point2f	right_landmark = points_vec[44];
		Point2f	right_landmark_down = points_vec[27];

		Point2f	endPt = points_vec[30];

		//# �����4���㵽��6����ľ�����Ϊ����
		//float r_left = sqrt(
		//	(left_landmark.x - left_landmark_down.x) * (left_landmark.x - left_landmark_down.x) +
		//	(left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));
		//cout << "���۾���:" << r_left;
		float r_left = b;

		//	# �����14���㵽��16����ľ�����Ϊ����
		//float	r_right = sqrt(
		//	(right_landmark.x - right_landmark_down.x) * (right_landmark.x - right_landmark_down.x) +
		//	(right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y));
		//cout << "���۾���:" << r_right;
		float r_right = b;
		//	# ����                     
		m_pIntance->LocalTranslationWarp_Eye(src, dst, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
		//	# ����
		m_pIntance->LocalTranslationWarp_Eye(src, dst, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);

	}
	imshow("Beauty", dst);
}


//ͼ��ֲ������㷨
void Beauty::LocalTranslationWarp_Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius)
{
	//ƽ�ƾ��� 
	float ddradius = radius * radius;
	//����|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//���� ͼ��ĸ�  �� ͨ������
	int height = img.rows;
	int width = img.cols;
	int chan = img.channels();

	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// # ����õ��Ƿ����α�Բ�ķ�Χ֮��
			//# �Ż�����һ����ֱ���ж��ǻ��ڣ�startX, startY)�ľ������
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				float rnorm = sqrt(distance) / radius;
				float ratio = 1 - (rnorm - 1)*(rnorm - 1)*0.5;
				//ӳ��ԭλ��
				float UX = warpX + ratio * (i - warpX);
				float UY = warpY + ratio * (j - warpY);

				//����˫���Բ�ֵ�õ�UX UY��ֵ
				BilinearInsert(img, dst, UX, UY, i, j);
			}
		}
	}
}

void Beauty::BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j)
{
	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	int c = src.channels();
	if (c == 3)
	{
		//�洢ͼ��ø�������
		CvPoint2D32f uv;
		CvPoint3D32f f1;
		CvPoint3D32f f2;

		//ȡ����
		int iu = (int)ux;
		int iv = (int)uy;
		uv.x = iu + 1;
		uv.y = iv + 1;

		//stepͼ�������е�ʵ�ʿ��  ����ͨ�����м���(0 , 1 2  ��ͨ��)
		f1.x = ((uchar*)(src.data + src.step*iv))[iu * 3] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3] * (uv.x - iu);
		f1.y = ((uchar*)(src.data + src.step*iv))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 1] * (uv.x - iu);
		f1.z = ((uchar*)(src.data + src.step*iv))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 2] * (uv.x - iu);


		f2.x = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3] * (uv.x - iu);
		f2.y = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 1] * (uv.x - iu);
		f2.z = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 2] * (uv.x - iu);

		((uchar*)(dst.data + dst.step*j))[i * 3] = f1.x*(1 - Abs(uv.y - iv)) + f2.x*(Abs(uv.y - iv));  //����ͨ�����и�ֵ
		((uchar*)(dst.data + dst.step*j))[i * 3 + 1] = f1.y*(1 - Abs(uv.y - iv)) + f2.y*(Abs(uv.y - iv));
		((uchar*)(dst.data + dst.step*j))[i * 3 + 2] = f1.z*(1 - Abs(uv.y - iv)) + f2.z*(Abs(uv.y - iv));

	}
}

//����Ч������
void Beauty::on_thinFace(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat dst = src.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f endPt = points_vec[34];
		for (int i = 3; i < 15; i = i + 2)
		{
			Point2f start_landmark = points_vec[i];
			Point2f end_landmark = points_vec[i + 2];

			//�����������루��������������룩
			//float dis = sqrt(
			//	(start_landmark.x - end_landmark.x) * (start_landmark.x - end_landmark.x) +
			//	(start_landmark.y - end_landmark.y) * (start_landmark.y - end_landmark.y));
			float dis = b;
			dst = m_pIntance->LocalTranslationWarp_Face(dst, start_landmark.x, start_landmark.y, endPt.x, endPt.y, dis);
		}
	}
	imshow("Beauty", dst);
}

Mat Beauty::LocalTranslationWarp_Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius)
{
	Mat dst = img.clone();
	//ƽ�ƾ��� 
	float ddradius = radius * radius;
	//����|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//���� ͼ��ĸ�  �� ͨ������
	int height = img.rows;
	int width = img.cols;
	int chan = img.channels();

	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};


	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// # ����õ��Ƿ����α�Բ�ķ�Χ֮��
			//# �Ż�����һ����ֱ���ж��ǻ��ڣ�startX, startY)�ľ������
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				//# �������i, j�������ԭ����
				//# ���㹫ʽ���ұ�ƽ������Ĳ���
				float ratio = (ddradius - distance) / (ddradius - distance + mc);
				ratio *= ratio;

				//ӳ��ԭλ��
				float UX = i - ratio * (endX - warpX);
				float UY = j - ratio * (endY - warpY);

				//����˫���Բ�ֵ�õ�UX UY��ֵ
				BilinearInsert(img, dst, UX, UY, i, j);
				//�ı䵱ǰ��ֵ
			}
		}
	}

	return dst;

}

void Beauty::on_beautyFace(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat img = src.clone();
	double scale = 1.3;

	CascadeClassifier cascade = m_pIntance->loadCascadeClassifier("D:\\opencv\\opencv-4.5.1-vc14_vc15\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");//������ѵ������
	CascadeClassifier netcascade = m_pIntance->loadCascadeClassifier("D:\\opencv\\opencv-4.5.1-vc14_vc15\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");//���۵�ѵ������
	if (cascade.empty() || netcascade.empty())
		return;
	m_pIntance->detectAndDraw(img, cascade, scale, b);
	if (m_pIntance->isDetected == false)
	{
		cout << "enter" << endl;
		Mat dst;

		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //˫���˲�����֮һ  
		//double fc = value1 * 12.5; //˫���˲�����֮һ  
		double fc = b;
		int p = 50;//͸����  
		Mat temp1, temp2, temp3, temp4;

		//��ԭͼ��image����˫���˲����������temp1ͼ����
		bilateralFilter(img, temp1, dx, fc, fc);

		//��temp1ͼ���ȥԭͼ��image�����������temp2ͼ����
		temp2 = (temp1 - img + 128);

		//��˹ģ��  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

		//��ԭͼ��imageΪ��ɫ����temp3ͼ��Ϊ���ɫ��������ͼ��������Թ��ϵõ�ͼ��temp4
		temp4 = img + 2 * temp3 - 255;

		//���ǲ�͸���ȣ�������һ���Ľ�����õ�����ͼ��dst
		dst = (img*(100 - p) + temp4 * p) / 100;
		dst.copyTo(img);
	}
	imshow("Beauty", img);
}

CascadeClassifier Beauty::loadCascadeClassifier(const string cascadePath)
{
	CascadeClassifier cascade;
	if (!cascadePath.empty())
	{
		if (!cascade.load(cascadePath))//��ָ�����ļ�Ŀ¼�м��ؼ���������
		{
			cerr << "ERROR: Could not load classifier cascade" << endl;
		}
	}
	return cascade;
}

void Beauty::detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, int val)
{
	std::vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };//�ò�ͬ����ɫ��ʾ��ͬ������
	//��ͼƬ��С���ӿ����ٶ�
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	//��Ϊ�õ�����haar���������Զ��ǻ��ڻҶ�ͼ��ģ�����Ҫת���ɻҶ�ͼ��
	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//���ߴ���С��1/scale,�����Բ�ֵ
	equalizeHist(smallImg, smallImg);//ֱ��ͼ����
	cascade.detectMultiScale(smallImg, //image��ʾ����Ҫ��������ͼ��
		faces,//objects��ʾ��⵽������Ŀ������
		1.1, //caleFactor��ʾÿ��ͼ��ߴ��С�ı���
		2, //minNeighbors��ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�С�����Լ�⵽����),
		0 | CASCADE_SCALE_IMAGE,//minSizeΪĿ�����С�ߴ�
		Size(30, 30)); //minSizeΪĿ������ߴ�
	int i = 0;
	//�������ľ��ο�
	for (std::vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		isDetected = true;
		Mat smallImgROI;
		std::vector<Rect> nestedObjects;
		Point center, left, right;
		Scalar color = colors[i % 8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);//��ԭ��ԭ���Ĵ�С
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);

		left.x = center.x - radius;
		left.y = cvRound(center.y - radius * 1.3);

		if (left.y < 0)
		{
			left.y = 0;
		}
		right.x = center.x + radius;
		right.y = cvRound(center.y + radius * 1.3);

		if (right.y > img.rows)
		{
			right.y = img.rows;
		}
		//ԭ���㷨
		//����-ĥƤ�㷨
		//Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
		
		//�滭ʶ���������
		//rectangle(img, left, right, Scalar(255, 0, 0));
		Mat roi = img(Range(left.y, right.y), Range(left.x, right.x));

		Mat dst;
		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //˫���˲�����֮һ  
		//double fc = value1 * 12.5; //˫���˲�����֮һ 
		double fc = val;//�仯ֵ
		int p = 50;//͸����  
		Mat temp1, temp2, temp3, temp4;

		//˫���˲�    ����ͼ�� ���ͼ�� ÿ���������ֱ����Χ��ɫ�ռ��������sigma  ����ռ��˲�����sigma 
		bilateralFilter(roi, temp1, dx, fc, fc);
		temp2 = (temp1 - roi + 128);
		//��˹ģ��  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
		temp4 = roi + 2 * temp3 - 255;
		dst = (roi*(100 - p) + temp4 * p) / 100;
		dst.copyTo(roi);
	}
}

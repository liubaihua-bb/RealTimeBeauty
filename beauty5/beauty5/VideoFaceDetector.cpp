#include "VideoFaceDetector.h"

VideoFaceDetector* VideoFaceDetector::m_pIntance = nullptr;//�����ⶨ�壬��cpp�ж���

const double VideoFaceDetector::TICK_FREQUENCY = getTickFrequency();

VideoFaceDetector::VideoFaceDetector(VideoCapture &videoCapture)
{
	m_videoCapture = &videoCapture;
	//��ʼ����̬����
	m_pIntance = this;//��ָ��ָ��ö���
}

bool VideoFaceDetector::isFaceFound() const
{
	return m_foundFace;
}

//���������϶���>>����ȡ��Ƶ֡�����м��
void VideoFaceDetector::operator>>(Mat &frame)
{
	//����Ƶ֡��ȡ��Mat������
	*m_videoCapture >> frame;
}

void VideoFaceDetector::beauty(void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	addWeighted(img, 1.2, m, 0, 50, dst);
	//1.2:�Աȶ� 50������

	m_pIntance->m_vecFaceData = m_pIntance->detectFace68(dst);

	//����
	double big = 20;
	Mat dst1 = dst.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f left_landmark = points_vec[38];
		Point2f	left_landmark_down = points_vec[27];

		Point2f	right_landmark = points_vec[44];
		Point2f	right_landmark_down = points_vec[27];

		Point2f	endPt = points_vec[30];

		float r_left = big;
		float r_right = big;
		//	# ����                     
		m_pIntance->Eye(dst, dst1, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
		//	# ����
		m_pIntance->Eye(dst, dst1, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);

	}

	//����
	double thin = 50;
	Mat dst2 = dst1.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f endPt = points_vec[34];
		for (int i = 3; i < 15; i = i + 2)
		{
			Point2f start_landmark = points_vec[i];
			Point2f end_landmark = points_vec[i + 2];
			float dis = thin;
			dst2 = m_pIntance->Face(dst2, start_landmark.x, start_landmark.y, endPt.x, endPt.y, dis);
		}
	}

	//ĥƤ
	double smooth = 20;
	Mat dst3 = dst2.clone();
	double scale = 1.3;

	CascadeClassifier cascade = m_pIntance->loadCascadeClassifier("D:\\opencv\\opencv-4.5.1-vc14_vc15\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");//������ѵ������
	CascadeClassifier netcascade = m_pIntance->loadCascadeClassifier("D:\\opencv\\opencv-4.5.1-vc14_vc15\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");//���۵�ѵ������
	if (cascade.empty() || netcascade.empty())
		return;
	m_pIntance->detectAndDraw(dst3, cascade, scale, smooth);
	if (m_pIntance->m_foundFace == false)
	{
		cout << "enter" << endl;
		Mat ddst;

		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //˫���˲�����֮һ  
		//double fc = value1 * 12.5; //˫���˲�����֮һ  
		double fc = smooth;
		int p = 50;//͸����  
		Mat temp1, temp2, temp3, temp4;

		//��ԭͼ��image����˫���˲����������temp1ͼ����
		bilateralFilter(dst3, temp1, dx, fc, fc);

		//��temp1ͼ���ȥԭͼ��image�����������temp2ͼ����
		temp2 = (temp1 - dst3 + 128);

		//��˹ģ��  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

		//��ԭͼ��imageΪ��ɫ����temp3ͼ��Ϊ���ɫ��������ͼ��������Թ��ϵõ�ͼ��temp4
		temp4 = dst3 + 2 * temp3 - 255;

		//���ǲ�͸���ȣ�������һ���Ľ�����õ�����ͼ��dst
		ddst = (dst3*(100 - p) + temp4 * p) / 100;
		ddst.copyTo(dst3);
	}


	imshow("Beauty", dst3);
}
//�������ݼ�� ͨ������������dlib����ȡ68�����Ե����ݵ�
std::vector<std::vector<Point2f>> VideoFaceDetector::detectFace68(Mat dst)
{
	std::vector<std::vector<Point2f>> rets;
	//����ͼƬ·��
	cv_image<bgr_pixel> cimg(dst);
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
void VideoFaceDetector::Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius)
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

void VideoFaceDetector::BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j)
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

		//stepͼ�������е�ʵ�ʿ��  ����ͨ�����м���(0 1 2  ��ͨ��)
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

Mat VideoFaceDetector::Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius)
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

CascadeClassifier VideoFaceDetector::loadCascadeClassifier(const string cascadePath)
{
	CascadeClassifier cascade;
	if (!cascadePath.empty())
	{
		if (!cascade.load(cascadePath))//��ָ�����ļ�Ŀ¼�м��ؼ���������
		{
			cerr << "ERROR: Could not load classifier cascade!" << endl;
		}
	}
	return cascade;
}

void VideoFaceDetector::detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, int val)
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
	//ת���ɻҶ�ͼ��
	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//���ߴ���С��1/scale,�����Բ�ֵ
	equalizeHist(smallImg, smallImg);//ֱ��ͼ����
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30)); 
	int i = 0;
	//�������ľ��ο�
	for (std::vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		m_foundFace = true;
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

		Mat roi = img(Range(left.y, right.y), Range(left.x, right.x));

		Mat dst;
		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //˫���˲�����֮һ  
		double fc = val;//�仯ֵ
		int p = 50;//͸����  
		Mat temp1, temp2, temp3, temp4;

		//˫���˲�
		bilateralFilter(roi, temp1, dx, fc, fc);
		temp2 = (temp1 - roi + 128);
		//��˹ģ��  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
		temp4 = roi + 2 * temp3 - 255;
		dst = (roi*(100 - p) + temp4 * p) / 100;
		dst.copyTo(roi);
	}
}


#include "VideoFaceDetector.h"

VideoFaceDetector* VideoFaceDetector::m_pIntance = nullptr;//在类外定义，在cpp中定义

const double VideoFaceDetector::TICK_FREQUENCY = getTickFrequency();

VideoFaceDetector::VideoFaceDetector(VideoCapture &videoCapture)
{
	m_videoCapture = &videoCapture;
	//初始化静态对象
	m_pIntance = this;//令指针指向该对象
}

bool VideoFaceDetector::isFaceFound() const
{
	return m_foundFace;
}

//重新在类上定义>>，获取视频帧并进行检测
void VideoFaceDetector::operator>>(Mat &frame)
{
	//将视频帧读取到Mat矩阵中
	*m_videoCapture >> frame;
}

void VideoFaceDetector::beauty(void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	addWeighted(img, 1.2, m, 0, 50, dst);
	//1.2:对比度 50：亮度

	m_pIntance->m_vecFaceData = m_pIntance->detectFace68(dst);

	//大眼
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
		//	# 瘦左                     
		m_pIntance->Eye(dst, dst1, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
		//	# 瘦右
		m_pIntance->Eye(dst, dst1, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);

	}

	//瘦脸
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

	//磨皮
	double smooth = 20;
	Mat dst3 = dst2.clone();
	double scale = 1.3;

	CascadeClassifier cascade = m_pIntance->loadCascadeClassifier("D:\\opencv\\opencv-4.5.1-vc14_vc15\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");//人脸的训练数据
	CascadeClassifier netcascade = m_pIntance->loadCascadeClassifier("D:\\opencv\\opencv-4.5.1-vc14_vc15\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");//人眼的训练数据
	if (cascade.empty() || netcascade.empty())
		return;
	m_pIntance->detectAndDraw(dst3, cascade, scale, smooth);
	if (m_pIntance->m_foundFace == false)
	{
		cout << "enter" << endl;
		Mat ddst;

		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //双边滤波参数之一  
		//double fc = value1 * 12.5; //双边滤波参数之一  
		double fc = smooth;
		int p = 50;//透明度  
		Mat temp1, temp2, temp3, temp4;

		//对原图层image进行双边滤波，结果存入temp1图层中
		bilateralFilter(dst3, temp1, dx, fc, fc);

		//将temp1图层减去原图层image，将结果存入temp2图层中
		temp2 = (temp1 - dst3 + 128);

		//高斯模糊  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

		//以原图层image为基色，以temp3图层为混合色，将两个图层进行线性光混合得到图层temp4
		temp4 = dst3 + 2 * temp3 - 255;

		//考虑不透明度，修正上一步的结果，得到最终图像dst
		ddst = (dst3*(100 - p) + temp4 * p) / 100;
		ddst.copyTo(dst3);
	}


	imshow("Beauty", dst3);
}
//人脸数据检测 通过调用三方库dlib来获取68点特性点数据的
std::vector<std::vector<Point2f>> VideoFaceDetector::detectFace68(Mat dst)
{
	std::vector<std::vector<Point2f>> rets;
	//加载图片路径
	cv_image<bgr_pixel> cimg(dst);
	//定义人脸检测器
	frontal_face_detector detector = get_frontal_face_detector();
	std::vector<dlib::rectangle> dets = detector(cimg);

	for (auto var : dets)
	{
		//关键点检测器
		shape_predictor sp;
		deserialize("D:\\dlib19.22\\dlib-19.22\\shape_predictor_68_face_landmarks.dat") >> sp;
		//定义shape对象保存检测的68个关键点
		full_object_detection shape = sp(cimg, var);
		//存储文件
		ofstream out("face_detector.txt");
		//读取关键点到容器中
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
	cout << "人脸检测结束:" << dets.size() << "张人脸数据" << endl;
	return rets;
}
void VideoFaceDetector::Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius)
{
	//平移距离 
	float ddradius = radius * radius;
	//计算|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//计算 图像的高  宽 通道数量
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
			// # 计算该点是否在形变圆的范围之内
			//# 优化，第一步，直接判断是会在（startX, startY)的矩阵框中
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				float rnorm = sqrt(distance) / radius;
				float ratio = 1 - (rnorm - 1)*(rnorm - 1)*0.5;
				//映射原位置
				float UX = warpX + ratio * (i - warpX);
				float UY = warpY + ratio * (j - warpY);

				//根据双线性插值得到UX UY的值
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
		//存储图像得浮点坐标
		CvPoint2D32f uv;
		CvPoint3D32f f1;
		CvPoint3D32f f2;

		//取整数
		int iu = (int)ux;
		int iv = (int)uy;
		uv.x = iu + 1;
		uv.y = iv + 1;

		//step图象像素行的实际宽度  三个通道进行计算(0 1 2  三通道)
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

		((uchar*)(dst.data + dst.step*j))[i * 3] = f1.x*(1 - Abs(uv.y - iv)) + f2.x*(Abs(uv.y - iv));  //三个通道进行赋值
		((uchar*)(dst.data + dst.step*j))[i * 3 + 1] = f1.y*(1 - Abs(uv.y - iv)) + f2.y*(Abs(uv.y - iv));
		((uchar*)(dst.data + dst.step*j))[i * 3 + 2] = f1.z*(1 - Abs(uv.y - iv)) + f2.z*(Abs(uv.y - iv));

	}
}

Mat VideoFaceDetector::Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius)
{
	Mat dst = img.clone();
	//平移距离 
	float ddradius = radius * radius;
	//计算|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//计算 图像的高  宽 通道数量
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
			// # 计算该点是否在形变圆的范围之内
			//# 优化，第一步，直接判断是会在（startX, startY)的矩阵框中
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				//# 计算出（i, j）坐标的原坐标
				//# 计算公式中右边平方号里的部分
				float ratio = (ddradius - distance) / (ddradius - distance + mc);
				ratio *= ratio;

				//映射原位置
				float UX = i - ratio * (endX - warpX);
				float UY = j - ratio * (endY - warpY);

				//根据双线性插值得到UX UY的值
				BilinearInsert(img, dst, UX, UY, i, j);
				//改变当前的值
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
		if (!cascade.load(cascadePath))//从指定的文件目录中加载级联分类器
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
		CV_RGB(255,0,255) };//用不同的颜色表示不同的人脸
	//将图片缩小，加快检测速度
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	//转换成灰度图像
	cvtColor(img, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//将尺寸缩小到1/scale,用线性插值
	equalizeHist(smallImg, smallImg);//直方图均衡
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30)); 
	int i = 0;
	//遍历检测的矩形框
	for (std::vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		m_foundFace = true;
		Mat smallImgROI;
		std::vector<Rect> nestedObjects;
		Point center, left, right;
		Scalar color = colors[i % 8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);//还原成原来的大小
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

		int dx = value1 * 5;    //双边滤波参数之一  
		double fc = val;//变化值
		int p = 50;//透明度  
		Mat temp1, temp2, temp3, temp4;

		//双边滤波
		bilateralFilter(roi, temp1, dx, fc, fc);
		temp2 = (temp1 - roi + 128);
		//高斯模糊  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
		temp4 = roi + 2 * temp3 - 255;
		dst = (roi*(100 - p) + temp4 * p) / 100;
		dst.copyTo(roi);
	}
}


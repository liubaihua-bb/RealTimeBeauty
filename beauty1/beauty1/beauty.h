#pragma once
#include "define.h"

/*class BeautyCam
{
public:
	//图像创建各个功能滑动条
	void initMainImgUI(const string&path);

	//对比度调节
	static void on_contrast(int b, void*userdata);

	//亮度调节
	static void on_lightness(int b, void*userdata);

	//眼睛调节
	static void on_BigEye(int b, void*userdata);

	//瘦脸效果
	//static void on_thinFace(int b, void*userdata);

	//美颜效果
	//static void on_beautyFace(int b, void*userdata);

	//人脸检测
	std::vector<std::vector<cv::Point2f>> dectectFace68(const string&);

	void LocalTranslationWarp_Eye(cv::Mat &, cv::Mat &, int, int, int, int, float);

	void BilinearInsert(cv::Mat &, cv::Mat &, float, float, int, int);

	//cv::Mat LocalTranslationWarp_Face(cv::Mat, float, float, float, float, float);

	//void detectAndDraw(cv::Mat&, cv::CascadeClassifier&, double, int);

	//cv::CascadeClassifier loadCascadeClassifier(const string& filename);

	cv::Mat m_MainImg;

	std::vector<std::vector<cv::Point2f>> m_vecFaceData;

	bool isDetected;

	static BeautyCam *m_pIntance;

	BeautyCam(){}
};*/

class Beauty
{
public:
	Beauty();
	void initMainImgUI();
	//static void browersBut_callback(int state, void *data);
	//对比度调节
	static void on_contrast(int b, void*userdata);
	//亮度调节
	static void on_lightness(int b, void*userdata);
	//眼睛调节
	static void on_BigEye(int b, void*userdata);
	//瘦脸效果
	static void on_thinFace(int b, void*userdata);
	//美颜效果
	static void on_beautyFace(int b, void*userdata);
	//提取人脸关键点
	std::vector<std::vector<Point2f>> dectectFace68(const string &path);
	//局部平移眼部放大
	void LocalTranslationWarp_Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius);
	//单个点线性插值
	void BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j);
	//局部平移脸部
	Mat LocalTranslationWarp_Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius);
private:
	//加载级联分类器
	CascadeClassifier loadCascadeClassifier(const string cascadePath);
	// 检测和绘制
	void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, int val);
private:
	Mat m_MainImg;
	static Beauty *m_pIntance;
	std::vector<std::vector<Point2f>> m_vecFaceData;
	bool isDetected = false;
};

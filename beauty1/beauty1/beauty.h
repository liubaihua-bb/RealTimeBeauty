#pragma once
#include "define.h"

/*class BeautyCam
{
public:
	//ͼ�񴴽��������ܻ�����
	void initMainImgUI(const string&path);

	//�Աȶȵ���
	static void on_contrast(int b, void*userdata);

	//���ȵ���
	static void on_lightness(int b, void*userdata);

	//�۾�����
	static void on_BigEye(int b, void*userdata);

	//����Ч��
	//static void on_thinFace(int b, void*userdata);

	//����Ч��
	//static void on_beautyFace(int b, void*userdata);

	//�������
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
	//�Աȶȵ���
	static void on_contrast(int b, void*userdata);
	//���ȵ���
	static void on_lightness(int b, void*userdata);
	//�۾�����
	static void on_BigEye(int b, void*userdata);
	//����Ч��
	static void on_thinFace(int b, void*userdata);
	//����Ч��
	static void on_beautyFace(int b, void*userdata);
	//��ȡ�����ؼ���
	std::vector<std::vector<Point2f>> dectectFace68(const string &path);
	//�ֲ�ƽ���۲��Ŵ�
	void LocalTranslationWarp_Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius);
	//���������Բ�ֵ
	void BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j);
	//�ֲ�ƽ������
	Mat LocalTranslationWarp_Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius);
private:
	//���ؼ���������
	CascadeClassifier loadCascadeClassifier(const string cascadePath);
	// ���ͻ���
	void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, int val);
private:
	Mat m_MainImg;
	static Beauty *m_pIntance;
	std::vector<std::vector<Point2f>> m_vecFaceData;
	bool isDetected = false;
};

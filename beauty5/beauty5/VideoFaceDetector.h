#pragma once
#include "define.h"

class VideoFaceDetector
{
public:
	VideoFaceDetector(VideoCapture &videoCapture);

	void								operator>>(Mat &frame);
	bool								isFaceFound() const;
	static void							beauty(void*userdata);
	std::vector<std::vector<Point2f>>	detectFace68(Mat);
	void								Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius);
	void								BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j);
	Mat									Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius);

private:
	static const double					TICK_FREQUENCY;
	static VideoFaceDetector*			m_pIntance;

	VideoCapture*						m_videoCapture = NULL;
	bool								m_foundFace = false;
	std::vector<std::vector<Point2f>>	m_vecFaceData;

	CascadeClassifier loadCascadeClassifier(const string cascadePath);
	void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, int val);
};


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include <iostream>

using namespace dlib;
using namespace std;


int main(int argc, char** argv)
{
	try
	{
		// This example takes in a shape model file and then a list of images to
		// process.  We will take these filenames in as command line arguments.
		// Dlib comes with example images in the examples/faces folder so give
		// those as arguments to this program.
		// 这个例子需要一个形状模型文件和一系列的图片.
//        if (argc == 1)
//        {
//            cout << "Call this program like this:" << endl;
//            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
//            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
//            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;//从这个地址下载模型标记点数据
//            return 0;
//        }

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		//****需要一个人脸检测器，获得一个边界框
		frontal_face_detector detector = get_frontal_face_detector();

		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		//****也需要一个形状预测器，这是一个工具用来预测给定的图片和脸边界框的标记点的位置。
		//****这里我们仅仅从shape_predictor_68_face_landmarks.dat文件加载模型
		shape_predictor sp;//定义个shape_predictor类的实例
		deserialize("D:\\dlib19.22\\dlib-19.22\\shape_predictor_68_face_landmarks.dat") >> sp;

		image_window win, win_faces;
		// Loop over all the images provided on the command line.
		// ****循环所有图片
//        for (int i = 2; i < argc; ++i)
		{
			//            cout << "processing image " << argv[i] << endl;
			array2d<rgb_pixel> img;//注意变量类型 rgb_pixel 三通道彩色图像
			load_image(img, "E:\\face\\face2.jfif");
			// Make the image larger so we can detect small faces.
			pyramid_up(img);

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			std::vector<rectangle> dets = detector(img);//检测人脸，获得边界框
			cout << "Number of faces detected: " << dets.size() << endl;//检测到人脸的数量

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			//****调用shape_predictor类函数，返回每张人脸的姿势
			std::vector<full_object_detection> shapes;//注意形状变量的类型，full_object_detection
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img, dets[j]);//预测姿势，注意输入是两个，一个是图片，另一个是从该图片检测到的边界框
				cout << "number of parts: " << shape.num_parts() << endl;
				//cout << "pixel position of first part:  " << shape.part(0) << endl;//获得第一个点的坐标,注意第一个点是从0开始的
				//cout << "pixel position of second part: " << shape.part(1) << endl;//获得第二个点的坐标
				//打印出全部68个点
				for (int i = 0; i < 68; i++)
				{
					cout << "第 " << i + 1 << " 个点的坐标： " << shape.part(i) << endl;
				}
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
			}

			// Now let's view our face poses on the screen.
			//**** 显示结果
			win.clear_overlay();
			win.set_image(img);
			win.add_overlay(render_face_detections(shapes));

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			//****我们也能提取每张剪裁后的人脸的副本，旋转和缩放到一个标准尺寸
			dlib::array<array2d<rgb_pixel> > face_chips;
			extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));

			cout << "Hit enter to process the next image..." << endl;
			cin.get();
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

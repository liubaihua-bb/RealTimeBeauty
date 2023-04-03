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
		// ���������Ҫһ����״ģ���ļ���һϵ�е�ͼƬ.
//        if (argc == 1)
//        {
//            cout << "Call this program like this:" << endl;
//            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
//            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
//            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;//�������ַ����ģ�ͱ�ǵ�����
//            return 0;
//        }

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		//****��Ҫһ����������������һ���߽��
		frontal_face_detector detector = get_frontal_face_detector();

		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		//****Ҳ��Ҫһ����״Ԥ����������һ����������Ԥ�������ͼƬ�����߽��ı�ǵ��λ�á�
		//****�������ǽ�����shape_predictor_68_face_landmarks.dat�ļ�����ģ��
		shape_predictor sp;//�����shape_predictor���ʵ��
		deserialize("D:\\dlib19.22\\dlib-19.22\\shape_predictor_68_face_landmarks.dat") >> sp;

		image_window win, win_faces;
		// Loop over all the images provided on the command line.
		// ****ѭ������ͼƬ
//        for (int i = 2; i < argc; ++i)
		{
			//            cout << "processing image " << argv[i] << endl;
			array2d<rgb_pixel> img;//ע��������� rgb_pixel ��ͨ����ɫͼ��
			load_image(img, "E:\\face\\face2.jfif");
			// Make the image larger so we can detect small faces.
			pyramid_up(img);

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.
			std::vector<rectangle> dets = detector(img);//�����������ñ߽��
			cout << "Number of faces detected: " << dets.size() << endl;//��⵽����������

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			//****����shape_predictor�ຯ��������ÿ������������
			std::vector<full_object_detection> shapes;//ע����״���������ͣ�full_object_detection
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img, dets[j]);//Ԥ�����ƣ�ע��������������һ����ͼƬ����һ���ǴӸ�ͼƬ��⵽�ı߽��
				cout << "number of parts: " << shape.num_parts() << endl;
				//cout << "pixel position of first part:  " << shape.part(0) << endl;//��õ�һ���������,ע���һ�����Ǵ�0��ʼ��
				//cout << "pixel position of second part: " << shape.part(1) << endl;//��õڶ����������
				//��ӡ��ȫ��68����
				for (int i = 0; i < 68; i++)
				{
					cout << "�� " << i + 1 << " ��������꣺ " << shape.part(i) << endl;
				}
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
			}

			// Now let's view our face poses on the screen.
			//**** ��ʾ���
			win.clear_overlay();
			win.set_image(img);
			win.add_overlay(render_face_detections(shapes));

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			//****����Ҳ����ȡÿ�ż��ú�������ĸ�������ת�����ŵ�һ����׼�ߴ�
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

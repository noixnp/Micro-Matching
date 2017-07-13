#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fftw3.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>    //BruteForceMatcheに必要。opencv2.4で移動した？
#include <functional>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")

#define IWIDTH 640
#define IHEIGHT 480
int size = 300;

int ux = IWIDTH / 2, uy = IHEIGHT / 2;

void resize(cv::Mat &img, double size){
	cv::resize(img, img, cv::Size(), size, size);
}

void toGrayscale(cv::Mat &img){
	cv::cvtColor(img, img, CV_RGB2GRAY);
}

void toColor(cv::Mat &img){
	cv::cvtColor(img, img, CV_GRAY2RGB);
}

// imgはグレースケール画像であること
void equalizeHist(cv::Mat &img){
	cv::equalizeHist(img, img);
}

void adaptiveThreshold(cv::Mat &img,int range){
	cv::adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, range, 0);
}

void transformImage(cv::Mat &img){
	//resize(img, 0.25);
	//resize(img, 4);
	toGrayscale(img);
	equalizeHist(img);
	adaptiveThreshold(img,25);
	toColor(img);
}

cv::Point2f clickedPos_aimg[4], clickedPos_timg[4];
cv::Point2f centerPos_aimg, centerPos_timg;
cv::Mat homography_authentication_img;

bool initMode = true;
int target = 0;

// 実装の簡素化と効率化のために，好ましくないやり方をしているので注意

double compaire_pair(cv::Mat &timg, cv::Mat &aimg){

	cv::Mat ctimg = timg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));
	cv::Mat caimg = aimg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));

	cvtColor(ctimg, ctimg, CV_RGB2GRAY);
	cvtColor(caimg, caimg, CV_RGB2GRAY);
	
	cv::Mat result;
	double max_val, min_val;
	cv::Point min_loc, max_loc;

	cv::matchTemplate(caimg, ctimg, result, CV_TM_CCORR_NORMED);

	//cout << result << endl;

	cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

	return max_val;
}

cv::Point2f squarePoints[] = {
	cv::Point2f((IWIDTH - size) / 2, (IHEIGHT - size) / 2),
	cv::Point2f((IWIDTH - size) / 2, (IHEIGHT + size) / 2),
	cv::Point2f((IWIDTH + size) / 2, (IHEIGHT + size) / 2),
	cv::Point2f((IWIDTH + size) / 2, (IHEIGHT - size) / 2)
};

void transformByAffine(cv::Mat &data, cv::Mat &data2, int lx, int ly, int angle){
	if (lx != 0 || ly != 0){
		// 移動
		cv::Mat mMat = (cv::Mat_<double>(2, 3) << 1.0, 0.0, lx, 0.0, 1.0, ly);
		cv::warpAffine(data, data2, mMat, data.size());
	}
	if (angle != 0){
		// 回転
		cv::Point2f center(data.cols*0.5, data.rows*0.5);
		cv::Mat affine_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::warpAffine(data, data2, affine_matrix, data.size());
	}
	cv::imshow("aaa", data2);
	cv::waitKey(0);
}

void perspective(cv::Mat data, cv::Point2f pts1[4], cv::Point2f pts2[4]){
	cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
	cv::warpPerspective(data, data, perspective_matrix, data.size(), cv::INTER_LINEAR);
}


string templateImgName[100];
string authenticationImgName[1000];

int threshold = 0x60;

void main2(){
	
	cv::Mat src = cv::imread("kuro/test.bmp");

	toGrayscale(src);

	cv::imwrite("test.bmp", src);

	cv::Mat out = src.clone();

	for (int y = 0; y < src.rows; ++y){
		for (int x = 0; x < src.cols; ++x){
			if (src.data[y * src.step + x * src.elemSize()] <= threshold)

				printf("(%d,%d)", x, y);

			int s = 1;
			while (true){
				int sr, sb, sg, k, minVal;
				sr = sb = sg = k = 0;
				for (int my = -s; my <= s; my++){
					for (int mx = -s; mx <= s; mx++){
						if (0 <= x + mx && x + mx < src.cols && 0 <= y + my && y + my < src.rows){
							if (src.data[(y + my) * src.step + (x + mx) * src.elemSize()] > threshold){
								sg += src.data[(y + my) * src.step + (x + mx) * src.elemSize()];
								k++;
							}
						}
					}
				}
				if (k != 0){
					out.data[y * src.step + x * src.elemSize()] = sg / k;
					out.data[y * src.step + x * src.elemSize() + 1] = sg / k;
					out.data[y * src.step + x * src.elemSize() + 2] = sg / k;
					break;
				}

				s++;
			}
		}
	}

	cv::imwrite("test.bmp", out);

	cv::imshow("test", out);
	cv::waitKey(0);


}

struct TemplateMatchingAnswer{
	double max_val;
	cv::Point2f max_loc;
	int angle;
};

TemplateMatchingAnswer templateMatching(cv::Mat timg, cv::Mat aimg){
	TemplateMatchingAnswer answer;
	answer.max_val = -1;

	for (int angle = -10; angle <= 10; angle += 2){
		cv::Mat _aimg;

		//cv::imshow("test", aimg);
		//cv::waitKey(0);

		cv::Point2f center(aimg.cols*0.5, aimg.rows*0.5);
		cv::Mat affine_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::warpAffine(aimg, _aimg, affine_matrix, aimg.size());

		cv::Mat result;
		double max_val, min_val;
		cv::Point min_loc, max_loc;

		cv::matchTemplate(timg, _aimg, result, CV_TM_CCORR_NORMED);
		cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

		if (max_val > answer.max_val){
			answer.max_val = max_val;
			answer.max_loc = max_loc;
			answer.angle = angle;
		}
	}

	return answer;
}

string templateImgSubName[1000];
string authenticationImgSubName[1000];

int main(){
	ifstream template_fp("template.txt");
	ifstream authentication_fp("authentication.txt");
	ofstream ofp("output.txt");

	cv::Point2f pts1[4];

	int i = 0;

	double angle;
	double mx, my;

	int tempN = 0, authN=0;
	while (template_fp >> templateImgName[tempN]){
		templateImgSubName[tempN] = templateImgName[tempN];
		templateImgName[tempN] = "template/" + templateImgName[tempN];
		tempN++;
	}
	while (authentication_fp >> authenticationImgName[authN]){
		authenticationImgSubName[authN] = authenticationImgName[authN];
		authenticationImgName[authN] = "authentication/" + authenticationImgName[authN];
		authN++;
	}

	for (int i = 0; i < tempN; i++){

		cv::Mat timg = cv::imread(templateImgName[i]);
		cv::Mat ctimg = timg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));
		transformImage(ctimg);

		string tofilename = "output_test_template/";
		tofilename += templateImgSubName[i];
		cv::imwrite(tofilename.c_str(), ctimg);

		for (int j = 0; j < authN; j++){
			cout << templateImgName[i] << "," << authenticationImgName[j] << endl;

			cv::Mat aimg = cv::imread(authenticationImgName[j]);
			transformImage(aimg);

			TemplateMatchingAnswer answer = templateMatching(ctimg, aimg);

			cv::Point2f center(aimg.cols*0.5, aimg.rows*0.5);
			cv::Mat affine_matrix = cv::getRotationMatrix2D(center, answer.angle, 1.0);
			cv::warpAffine(aimg, aimg, affine_matrix, aimg.size());

			cv::Point2f pts[4];
			pts[0].x = answer.max_loc.x;
			pts[0].y = answer.max_loc.y;

			pts[1].x = answer.max_loc.x;
			pts[1].y = answer.max_loc.y + 256;

			pts[2].x = answer.max_loc.x + 256;
			pts[2].y = answer.max_loc.y + 256;

			pts[3].x = answer.max_loc.x + 256;
			pts[3].y = answer.max_loc.y;

			cv::line(aimg, pts[0], pts[1], cv::Scalar(255, 0, 0), 2, CV_AA);
			cv::line(aimg, pts[1], pts[2], cv::Scalar(255, 0, 0), 2, CV_AA);
			cv::line(aimg, pts[2], pts[3], cv::Scalar(255, 0, 0), 2, CV_AA);
			cv::line(aimg, pts[3], pts[0], cv::Scalar(255, 0, 0), 2, CV_AA);

			string aofilename = "output_test/";
			aofilename += templateImgSubName[i];
			aofilename += "_";
			aofilename += authenticationImgSubName[j];
			aofilename += ".png";
			cv::imwrite(aofilename.c_str(), aimg);
			
			ofp << templateImgName[i] << "," << authenticationImgName[j] << "," << answer.angle << "," << answer.max_loc << "," << answer.max_val << endl;
		}
	}
	puts("end");
	while (1);

}

int main22(){

	ifstream ifp("input.txt");
	ofstream ofp("output.txt");

	cv::Point2f pts1[4];

	int i = 0;

	double angle;
	double mx, my;

	while (ifp >> templateImgName[i] >> authenticationImgName[i]){
		ifp >> angle >> mx >> my;
		ifp >> pts1[0].x >> pts1[0].y
			>> pts1[1].x >> pts1[1].y
			>> pts1[2].x >> pts1[2].y
			>> pts1[3].x >> pts1[3].y;

		cv::Mat timg = cv::imread(templateImgName[i]);
		cv::Mat aimg = cv::imread(authenticationImgName[i]);

		transformByAffine(aimg, aimg, mx, my, angle);
		perspective(aimg, pts1, squarePoints);

		cv::Mat ctimg = timg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));
		cv::Mat caimg = aimg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));

		cv::Mat result;
		double max_val, min_val;
		cv::Point min_loc, max_loc;

		transformImage(ctimg);
		transformImage(caimg);

		cv::matchTemplate(caimg, ctimg, result, CV_TM_CCOEFF_NORMED);

		cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

		string a1, a2;
		a1 = "output/";
		a2 = "output/";

		cv::imwrite(a1+templateImgName[i], caimg);
		cv::imwrite(a2+authenticationImgName[i], ctimg);

		ofp << templateImgName[i] << "," << authenticationImgName[i] << "," << max_val << endl;
		i++;
	}

	for (int i = 0; i < 30; i++){
		for (int j = i + 1; j < 30; j++){
			cv::Mat timg = cv::imread(templateImgName[i]);
			cv::Mat aimg = cv::imread(templateImgName[j]);

			cv::Mat ctimg = timg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));
			cv::Mat caimg = aimg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size));

			cv::Mat result;
			double max_val, min_val;
			cv::Point min_loc, max_loc;

			transformImage(ctimg);
			transformImage(caimg);

			cv::matchTemplate(caimg, ctimg, result, CV_TM_CCOEFF_NORMED);

			cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

			string a1, a2;
			a1 = "output/";
			a2 = "output/";

			ofp << templateImgName[i] << "," << templateImgName[j] << "," << max_val << endl;
		}

	}

	ofp.close();

	puts("end");

	while (1);
}
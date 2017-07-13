#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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


#define IWIDTH 2592
#define IHEIGHT 1944
int size = 1600;

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

void adaptiveThreshold(cv::Mat &img, int range){
	cv::adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, range, 0);
}

void transformImage(cv::Mat &img){
	//resize(img, 0.25);
	//resize(img, 4);
	toGrayscale(img);
	equalizeHist(img);
	adaptiveThreshold(img, 25);
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

cv::Mat perspective(cv::Mat &data, cv::Point2f pts1[4], cv::Point2f pts2[4]){
	cv::Mat ret;
	cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
	cv::warpPerspective(data, ret, perspective_matrix, data.size(), cv::INTER_LINEAR);
	return ret;
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
	double xsize, ysize;
	int angle;
	cv::Point2f pts[4];
	cv::Point2f spts[4];
	int bidx;
	double n1, n2;
};

int z = 0;

cv::Point2f imagePoints[] = {
	cv::Point2f(0,0),
	cv::Point2f(0,IHEIGHT-1),
	cv::Point2f(IWIDTH-1,IHEIGHT-1),
	cv::Point2f(IWIDTH-1,0)
};

TemplateMatchingAnswer templateMatching(cv::Mat timg, cv::Mat aimg){
	TemplateMatchingAnswer answer;
	answer.max_val = -1;
	int angleN = 0;
	int sub = 0;
	cv::Mat _aimg, __aimg;
	cv::Mat result;
	double max_val, min_val;
	cv::Point min_loc, max_loc;

	cv::Point2f bPoints[4];

	cv::imwrite("output_test_template/test.png", timg);
	
	//　角度の決定
	for (double n1 = 0.9; n1 <= 1.11; n1 += 0.1){
		for (double n2 = 0.9; n2 <= 1.11; n2 += 0.1){
			printf("(%lf,%lf)\n", n1, n2);

			for (int i = 0; i < 4; i++){
				for (int j = 0; j < 4; j++){
					bPoints[j] = imagePoints[j];
					if (i == j){
						bPoints[j].x = imagePoints[j].x - (IWIDTH * (1 - n1));
						bPoints[j].y = imagePoints[j].y - (IHEIGHT * (1 - n1));
					}
					else if ((i+1)%2 == j){
						bPoints[j].x = imagePoints[j].x - (IWIDTH * (1 - n2));
						bPoints[j].y = imagePoints[j].y - (IHEIGHT * (1 - n2));
					}
				}
				cv::Mat prMat = perspective(aimg, bPoints, imagePoints);

				for (int angle = -12; angle <= 12; angle += 4){
					// cout << "angle: " << angle << "-> ";
					cv::Point2f center(aimg.cols*0.5, aimg.rows*0.5);
					cv::Mat affine_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
					cv::warpAffine(prMat, _aimg, affine_matrix, prMat.size());

					cv::matchTemplate(timg, _aimg, result, CV_TM_CCOEFF_NORMED);//マッチング
					cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);//ベストマッチを求める

					// cv::imwrite("ooo/" + to_string(sub++) + ".png", _aimg);

					// printf("%lf\n", max_val);

					if (max_val > answer.max_val){ //よりよいマッチを書き換え
						answer.max_val = max_val;
						answer.max_loc = max_loc;
						answer.angle = angle;
						answer.bidx = i;
						answer.n1 = n1;
						answer.n2 = n2;
					}
				}
				if (n1 == 1.0&&n2 == 1.0)break;
			}
		}
	}

	//透視変換
	for (int j = 0; j < 4; j++){
		bPoints[j] = imagePoints[j];
		if (answer.bidx == j){
			bPoints[j].x = imagePoints[j].x - (IWIDTH * (1 - answer.n1));
			bPoints[j].y = imagePoints[j].y - (IHEIGHT * (1 - answer.n1));
		}
		else if ((answer.bidx + 1) % 2 == j){
			bPoints[j].x = imagePoints[j].x - (IWIDTH * (1 - answer.n2));
			bPoints[j].y = imagePoints[j].y - (IHEIGHT * (1 - answer.n2));
		}
	}
	aimg = perspective(aimg, bPoints, imagePoints);

	// 回転だけした画像
	cv::Point2f center(aimg.cols*0.5, aimg.rows*0.5);
	cv::Mat affine_matrix = cv::getRotationMatrix2D(center, answer.angle, 1.0);
	cv::warpAffine(aimg, _aimg, affine_matrix, aimg.size());

	int range = 2;
	vector<cv::Point2f> zx;
	for (int mx = -range; mx <= range; mx += 1){
		for (int my = -range; my <= range; my += 1){
			zx.push_back(cv::Point2f(mx, my));
		}
	}

	printf("xsize : %lf, ysize: %lf, angle: %d\n", answer.xsize, answer.ysize, answer.angle); 

	cv::Point2f pts[4], spts[4], npts[4];

	pts[0].x = answer.max_loc.x;
	pts[0].y = answer.max_loc.y;
	pts[1].x = answer.max_loc.x;
	pts[1].y = answer.max_loc.y + size;
	pts[2].x = answer.max_loc.x + size;
	pts[2].y = answer.max_loc.y + size;
	pts[3].x = answer.max_loc.x + size;
	pts[3].y = answer.max_loc.y;

	spts[0].x = answer.max_loc.x;
	spts[0].y = answer.max_loc.y;
	spts[1].x = answer.max_loc.x;
	spts[1].y = answer.max_loc.y + size;
	spts[2].x = answer.max_loc.x + size;
	spts[2].y = answer.max_loc.y + size;
	spts[3].x = answer.max_loc.x + size;
	spts[3].y = answer.max_loc.y;

	npts[0].x = answer.max_loc.x;
	npts[0].y = answer.max_loc.y;
	npts[1].x = answer.max_loc.x;
	npts[1].y = answer.max_loc.y + size;
	npts[2].x = answer.max_loc.x + size;
	npts[2].y = answer.max_loc.y + size;
	npts[3].x = answer.max_loc.x + size;
	npts[3].y = answer.max_loc.y;


	bool contFlg = true;

	double tmax_val = answer.max_val;
	while (contFlg){
		contFlg = false;
		int nx, ny;
		for (int idx = 0; idx < 4; idx++){
			printf("search around %d: \n", idx); //ラウンド表示
			int phase = 0;
			while (true){
				printf("  phase %d\n", phase++);//フェイズ表示
				for (int zidx = 0; zidx < zx.size(); zidx++){
					npts[idx].x = pts[idx].x + zx[zidx].x;
					npts[idx].y = pts[idx].y + zx[zidx].y;
					cv::Mat prMat = perspective(_aimg, npts, spts);//ずらしつつ透視変換
					cv::Mat caimg = prMat(cv::Rect(answer.max_loc.x, answer.max_loc.y, size, size));//ベストマッチ点を抽出

					cv::matchTemplate(timg, caimg, result, CV_TM_CCOEFF_NORMED);

					cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
					if (max_val >= tmax_val){
						nx = zx[zidx].x; ny = zx[zidx].y;
						tmax_val = max_val;
					}
					//cv::imwrite("ooo/" + to_string(z++) + ".png", caimg);
				}
				printf("     (%d,%d): %lf\n", nx, ny, tmax_val); //スコア表示
				if (nx == 0 && ny == 0){
					break;
				}
				else{
					pts[idx].x += nx;
					pts[idx].y += ny;
					contFlg = true;
				}
			}
		}
	}

	if (answer.max_val <= tmax_val){
		answer.max_val = tmax_val;
		for (int idx = 0; idx < 4; idx++){
			answer.pts[idx] = pts[idx];
			answer.spts[idx] = spts[idx];
		}
	}

	return answer;
}

struct FILE_PAIR{
	string a;
	string asubname;
	string b;
	string bsubname;
};

FILE_PAIR files[10000];

int main(){
	ifstream ifs("inputdata.txt");
	ofstream ofp("output.txt");

	cv::Point2f bPoints[4];
	cv::Point2f fPoint1((IWIDTH - size)/2, (IHEIGHT - size)/2);
	cv::Point2f fPoint2((IWIDTH + size)/2, (IHEIGHT - size)/2);
	cv::Point2f fPoint3((IWIDTH - size)/2, (IHEIGHT + size)/2);
	cv::Point2f fPoint4((IWIDTH + size)/2, (IHEIGHT + size)/2);
	

	int N = 0;
	while (ifs >> files[N].asubname >> files[N].bsubname){
		files[N].a = "gazou/" + files[N].asubname;
		files[N].b = "gazou/" + files[N].bsubname;
		N++;
	}

	for (int i = 0; i < N; i++){
		cout << files[i].a << "," << files[i].b << endl;

		cv::Mat timg = cv::imread(files[i].a);
		transformImage(timg);

		cv::line(timg, fPoint1, fPoint2, cv::Scalar(255, 0, 0), 2, CV_AA); 
		cv::line(timg, fPoint1, fPoint3, cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(timg, fPoint2, fPoint4, cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(timg, fPoint3, fPoint4, cv::Scalar(255, 0, 0), 2, CV_AA);

		string tempfolder = "output_template/";
		tempfolder += files[i].asubname;
		tempfolder += ".png";
		cv::imwrite(tempfolder.c_str(), timg);

		
		// resize(timg, 0.5);
		cv::Mat ctimg = timg(cv::Rect((IWIDTH - size) / 2, (IHEIGHT - size) / 2, size, size)); //画像の真ん中辺りを抽出

		string aaofilename = "output_test_template/";
		aaofilename += files[i].asubname;
		aaofilename += ".png";
		cv::imwrite(aaofilename.c_str(), ctimg); //transformした画像を保存

		
		cv::Mat aimg = cv::imread(files[i].b);
		//resize(timg, 0.5);
		transformImage(aimg);

		TemplateMatchingAnswer answer = templateMatching(ctimg, aimg);


		for (int j = 0; j < 4; j++){
			bPoints[j] = imagePoints[j];
			if (answer.bidx == j){
				bPoints[j].x = imagePoints[j].x - (IWIDTH * (1 - answer.n1));
				bPoints[j].y = imagePoints[j].y - (IHEIGHT * (1 - answer.n1));
			}
			else if ((answer.bidx + 1) % 2 == j){
				bPoints[j].x = imagePoints[j].x - (IWIDTH * (1 - answer.n2));
				bPoints[j].y = imagePoints[j].y - (IHEIGHT * (1 - answer.n2));
			}
		}
		aimg = perspective(aimg, bPoints, imagePoints);//透視変換（画像をまっすぐ撮った風にする）


		// 回転だけした画像
		cv::Point2f center(aimg.cols*0.5, aimg.rows*0.5);
		cv::Mat affine_matrix = cv::getRotationMatrix2D(center, answer.angle, 1.0);//回転行列の生成
		cv::warpAffine(aimg, aimg, affine_matrix, aimg.size());


		cv::Mat transImg = perspective(aimg, answer.pts, answer.spts);//透視変換
		transImg = transImg(cv::Rect(answer.max_loc.x, answer.max_loc.y, size, size));
		string aofilename = "output_test_projective/";
		aofilename += files[i].asubname;
		aofilename += "_";
		aofilename += files[i].bsubname;
		aofilename += ".png";
		cv::imwrite(aofilename.c_str(), transImg);

		cv::line(aimg, answer.pts[0], answer.pts[1], cv::Scalar(255, 0, 0), 2, CV_AA); //ベストマッチ四角形書く
		cv::line(aimg, answer.pts[1], answer.pts[2], cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(aimg, answer.pts[2], answer.pts[3], cv::Scalar(255, 0, 0), 2, CV_AA);
		cv::line(aimg, answer.pts[3], answer.pts[0], cv::Scalar(255, 0, 0), 2, CV_AA);

		aofilename = "output_test/";
		aofilename += files[i].asubname;
		aofilename += "_";
		aofilename += files[i].bsubname;
		aofilename += ".png";
		cv::imwrite(aofilename.c_str(), aimg);

		ofp << files[i].a << "," <<
			files[i].b << "," << answer.angle << "," << answer.bidx << "," << answer.n1 << "," << answer.n2 << "," <<
			answer.spts[0] << "," << answer.spts[1] << "," << answer.spts[2] << "," << answer.spts[3] <<
			answer.pts[0] << "," << answer.pts[1] << "," << answer.pts[2] << "," << answer.pts[3] << "," << answer.max_val << endl;
			

	}



}


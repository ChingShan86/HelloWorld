#include <iostream>
#include <stdio.h>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

Point LeftTop(-1, -1), RightDown(-1, -1);
bool drawline = false;
const int len = 30;
vector<Point> q(len);
void onMouse(int Event, int x, int y, int flags, void* param) {
	if (Event == CV_EVENT_LBUTTONDOWN) {
		LeftTop = Point(-1, -1), RightDown = Point(-1, -1);
		LeftTop.x = x;
		LeftTop.y = y;
	}
	if (Event == CV_EVENT_LBUTTONUP) {
		RightDown.x = x;
		RightDown.y = y;
	}
	if (Event == CV_EVENT_RBUTTONDOWN) {
		drawline == false ? drawline = true : drawline = false;
	}
}
//膚色提取 
void SkinFilter1(Mat roi, Mat dst) {
	Mat blurred,hsv;
	blur(roi, blurred, Size(3, 3));
	cvtColor(blurred, hsv, CV_BGR2HSV);
	Mat  dst1, dst2, dst3, dst4, dst5,dst6,dst7;
	inRange(hsv, Scalar(0, 35/**15*/, 40/*越深值低*/), Scalar(10, 180/*若皮膚紅則調高至120否則為130左右*/, 230), dst1); //皮膚
	inRange(hsv, Scalar(10, 35/**15*/, 40/*越深值低*/), Scalar(25, 100, 150), dst2); //皮膚
	inRange(hsv, Scalar(10, 15, 40/*越深值低*/), Scalar(25, 130, 80), dst3); //皮膚
	inRange(hsv, Scalar(25, 25/**0亮環境*/, 30), Scalar(50, 100/*越深*/, 119), dst4); //手臂
	inRange(hsv, Scalar(25, 0, 110), Scalar(37, 45, 180), dst5); //肩膀、手心
	inRange(hsv, Scalar(160, 30/**5亮環境*/, 85), Scalar(190, 120/*若皮膚紅則調高至120否則為60左右*/, 250), dst6); //指尖
	inRange(hsv, Scalar(0, 15, 40), Scalar(50, 180, 250), dst7);
	Mat black, white;
	black.create(dst1.size(), dst1.type());  //不可用src因為是3通道 而dst1是單通道
	black = Scalar::all(0);
	white.create(dst1.size(), dst1.type());
	white = Scalar::all(255);
	black.copyTo(white, dst1);
	black.copyTo(white, dst2);
	black.copyTo(white, dst3);
	black.copyTo(white, dst4);
	black.copyTo(white, dst5);
	black.copyTo(white, dst6);
	black.copyTo(white, dst7);
	Mat thres;
	threshold(white, thres, 128, 255, THRESH_BINARY_INV);
	Mat dilation, erotion;
	dilate(thres, dilation, Mat(), Point(-1, -1), 2);
	erode(dilation, erotion, Mat(), Point(-1, -1), 2);
	erotion.copyTo(dst);
}
void SkinFilter2(Mat roi, Mat dst) {

	Mat gray;
	cvtColor(roi, gray, COLOR_BGR2GRAY);
	Mat blurred;
	GaussianBlur(gray, blurred, Size(3, 3), 0, 0);
	Mat thres1;
	//adaptiveThreshold(blurred,thres1,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 65, 0);
	Mat thres2;
	threshold(blurred, thres2, 140, 255, THRESH_BINARY_INV + THRESH_OTSU);
	Mat dilation, erotion;
	dilate(thres2, dilation, Mat(), Point(-1, -1), 2);
	erode(dilation, erotion, Mat(), Point(-1, -1), 2);
	erotion.copyTo(dst);
}
void SkinFilter3(Mat roi, Mat dst) {
	Mat gray;
	cvtColor(roi, gray, CV_BGR2GRAY);
	Mat blurred;
	GaussianBlur(gray, blurred, Size(3, 3), 0, 0);
	Mat canny;
	Canny(blurred, canny, 50, 85);
	Mat dilation, erotion;
	dilate(canny, dilation, Mat(), Point(-1, -1), 1);
	erode(dilation, erotion, Mat(), Point(-1, -1), 1);
	erotion.copyTo(dst);
}

// (1) 繪製手缺陷的 起始點、終點、最遠點
void DrawHandsDefect(Mat src, Mat roi, int i, vector<vector<Point> > contours_poly, vector<vector<Vec4i>> defect, int height, int* c) {
	vector<Vec4i>::iterator iter = defect[i].begin();
	if (iter == defect[i].end())
		(*c) = 0;
	else {
		while (iter != defect[i].end()) {
			Vec4i &v = (*iter);   ///從defect[i]複製出一個v
			int startidx = v[0];  ///start_index
			Point ptstart(contours_poly[i][startidx]);  ///point_start
			int endidx = v[1];
			Point ptend(contours_poly[i][endidx]);
			int faridx = v[2];
			Point ptfar(contours_poly[i][faridx]);
			int depth = v[3] / 256;
			//三角函數 兩向量求夾角
			double start_d = sqrt(pow(ptstart.x - ptfar.x, 2) + pow(ptstart.y - ptfar.y, 2));
			double end_d = sqrt(pow(ptend.x - ptfar.x, 2) + pow(ptend.y - ptfar.y, 2));
			int dot = (ptstart.x - ptfar.x)*(ptend.x - ptfar.x) + (ptstart.y - ptfar.y)*(ptend.y - ptfar.y);
			double cosine = dot / (start_d*end_d); ///小於80度
			if (cosine > 0.1736) {
				//	line(roi, ptstart, ptfar, Scalar(255, 255, 255), 2);
				//	line(roi, ptend, ptfar, Scalar(255, 255, 255), 2);
				circle(roi, ptstart, 5, Scalar(255, 0, 0), 2);
				circle(roi, ptend, 5, Scalar(0, 0, 0), 2);
				circle(roi, ptfar, 5, Scalar(0, 0, 255), 2);
				(*c)++;
			}
			iter++;
		}
	}
}
// (2) 手勢繪圖
void Drawlines(Mat roi, int x, int y, int width, int height) {
		int tip_x = x + 2, tip_y = y + 2;
		if (tip_x > 0 && tip_y > 0)
			q.insert(q.begin(), Point(tip_x, tip_y));
		for (int i = 0; i < len; i++) {
			if (i - 1 >= 0) {
				if (q[i - 1].x != 0 && q[i].x != 0)
					line(roi, q[i - 1], q[i], Scalar(0, 0, 0), (len - i), CV_AA);
			}
		}
		if (q[0].x != 0 && q[10].x != 0) {
			int delta_x = q[0].x - q[10].x, delta_y = q[0].y - q[10].y;
			if (delta_x > 30)       putText(roi, " Right", Point(100, 30), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
			else if (delta_x < -30)  putText(roi, " Left", Point(100, 30), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
			if (delta_y > 30)       putText(roi, " Down", Point(30, 30), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
			else if (delta_y < -30)  putText(roi, " Top", Point(30, 30), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
		}
}
// (2) 手勢資訊
void Gesture(Mat src, Mat roi, int i, vector<vector<Point> > contours_poly, vector<vector<Point> > hull) {
	double ratio1 = contourArea(Mat(contours_poly[i])/*,bool oriented = false*/) / contourArea(Mat(hull[i]));
	double ratio2 = arcLength(Mat(contours_poly[i]), 1) / arcLength(Mat(hull[i]), 1);
	double ratio3 = arcLength(Mat(contours_poly[i]), 1) / contourArea(Mat(contours_poly[i]));
	char info1[20], info2[20], info3[20];
	sprintf(info1, "area/Area: %.2f", ratio1);
	sprintf(info2, "len/Len: %.3f", ratio2);
	sprintf(info3, "area/len: %.4f", abs(ratio3));
	putText(src, info1, Point(10, 375), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	putText(src, info2, Point(10, 405), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	putText(src, info3, Point(10, 435), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	if (ratio2 >= 1 && ratio2 <= 1.045) {
		putText(roi, "fist", Point(10, 55), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	}
	else if (ratio2 >= 1.05&&ratio2 <= 1.1) {
		putText(roi, "one", Point(10, 55), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	}
	else if (ratio2 >= 1.2&&ratio2 <= 1.32) {
		putText(roi, "YA", Point(10, 55), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	}
	else if (ratio2 >= 1.44&&ratio2 <= 1.54) {
		putText(roi, "three", Point(10, 55), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	}
	else if (ratio2 >= 1.55&&ratio2 <= 1.67) {
		putText(roi, "four", Point(10, 55), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	}
	else if (ratio2 >= 1.68&&ratio2 <= 2) {
		putText(roi, "palm", Point(10, 55), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
	}
}

//找手的輪廓線
void HandsContours1(Mat src, Mat roi, Mat thres, int*count) {

	//影像中尋找所有的輪廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thres, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

	//輪廓多邊形化 以及 計算能夠包圍輪廓的長方形區域面積
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> contours_poly_boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);    //輪廓多邊形化
		contours_poly_boundRect[i] = boundingRect(Mat(contours_poly[i]));           //計算可以包含輪廓的最小長方形區域
	}

	//根據輪廓(多邊形化)找 凸殼(最外殼)、缺陷	
	vector<vector<Point> >hull(contours.size());
	vector<vector<int> >hulls(contours.size());
	vector<vector<Vec4i>> defect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours_poly[i]), hull[i], false);
		convexHull(Mat(contours_poly[i]), hulls[i], false);
		convexityDefects(Mat(contours_poly[i]), hulls[i], defect[i]);
	}

	vector<Rect> hull_boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		hull_boundRect[i] = boundingRect(Mat(hull[i]));
	}
	//繪製與手相關圖形 指尖、指縫
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	int x = -1, y = -1, width = -1, height = -1;
	for (int i = 0; i < contours_poly.size(); i++) {
		if (contours_poly_boundRect[i].area() > 10000) {

			mu[i] = moments(contours_poly[i], false);
			mc[i] = Point2f((mu[i].m10 / mu[i].m00), (mu[i].m01 / mu[i].m00));

			//drawContours(roi, hull, i, Scalar(0, 255, 0), 2, CV_AA, vector<Vec4i>(), 0, Point());
			rectangle(roi, contours_poly_boundRect[i], Scalar(255, 255, 0), 1, CV_AA);  //繪製包圍輪廓的最大長方形
			DrawHandsDefect(src, roi, i, contours_poly, defect, contours_poly_boundRect[i].height, count);

			x = contours_poly_boundRect[i].x, y = contours_poly_boundRect[i].y;
			width = contours_poly_boundRect[i].width, height = contours_poly_boundRect[i].height;
			Drawlines(roi, x, y, width, height);
		}
	}
}
void HandsContours2(Mat src, Mat roi, Mat thres, int*count) {
	//影像中尋找所有的輪廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thres, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	//輪廓多邊形化 以及 計算能夠包圍輪廓的長方形區域面積
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> contours_poly_boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);    //輪廓多邊形化
		contours_poly_boundRect[i] = boundingRect(Mat(contours_poly[i]));           //計算可以包含輪廓的最小長方形區域
	}
	//根據輪廓(多邊形化)找 凸殼(最外殼)、缺陷	
	vector<vector<Point> >hull(contours.size());
	vector<vector<int> >hulls(contours.size());
	vector<vector<Vec4i>> defect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours_poly[i]), hull[i], false);
		convexHull(Mat(contours_poly[i]), hulls[i], false);
		convexityDefects(Mat(contours_poly[i]), hulls[i], defect[i]);
	}
	//繪製與手相關圖形 指尖、指縫
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	int x = -1, y = -1, width = -1, height = -1;
	for (int i = 0; i < contours_poly.size(); i++) {
		if (contours_poly_boundRect[i].area() > 5000) {
			mu[i] = moments(contours_poly[i], false);
			mc[i] = Point2f((mu[i].m10 / mu[i].m00), (mu[i].m01 / mu[i].m00));
			drawContours(roi,hull, i, Scalar(0, 0, 255), 2, CV_AA, vector<Vec4i>(), 0, Point());
			rectangle(roi, contours_poly_boundRect[i], Scalar(255, 255, 0), 1, CV_AA);  //繪製包圍輪廓的最大長方形
			x = contours_poly_boundRect[i].x, y = contours_poly_boundRect[i].y;
			width = contours_poly_boundRect[i].width, height = contours_poly_boundRect[i].height;
			if (drawline == true) Drawlines(roi, x, y, width, height); //如果右鍵
			if (drawline == false) Gesture(src, roi, i, contours_poly, hull);
		}
	}
}

//---------------------------------------------------------------------------------------------------------------------------//

//主程式
int main() {
	VideoCapture sam(0);
	namedWindow("鏡頭影像", 1);
	setMouseCallback("鏡頭影像",onMouse,NULL);
	while (1) {
		Mat src;
		sam >> src;
		flip(src, src, 1);
		if (LeftTop.x == -1 || RightDown.x == -1)  imshow("鏡頭影像", src);	
		if (LeftTop.x != -1 && RightDown.x != -1) {
			int count = 1; //紀錄手比的是多少
			Mat roi;
			roi = src(Rect(LeftTop, RightDown));

			Mat thres(roi.size(), CV_8UC1);
			SkinFilter2(roi,thres);			
			HandsContours2(src,roi, thres, &count);

			//輸出手勢及移動方向的訊息
			//char buf[20];
		//	sprintf(buf, "%d", count);
		//	putText(roi, buf, Point(10, 30), 0, 0.8, Scalar(0, 0, 255), 2, CV_AA);
			//imshow("結果圖", roi);
			imshow("二值圖",thres);

			rectangle(src, Rect(LeftTop, RightDown), Scalar(0, 255, 255), 2, CV_AA);
			imshow("鏡頭影像", src);
		}
		if (waitKey(33) == 27)break;
	}
	return 0;
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Drawing;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using System.Windows.Forms;

namespace HandGesture_Recognition_Project
{
    class My_img_processing:My_Math
    {
        public Image<Hsv, byte> hsv_frame = null;
        public Image<Bgr, byte> bgr_frame = null;
        public Image<Gray, byte> imgGray = null;
        public Image<Gray, byte> imgBinary = null;
        public Image<Gray, byte> imgSkin = null;
        public Image<Gray, byte> t = null;


        /// Generate camera frame
        public Image<Hsv, byte> GenerateCameraFrame_hsv(Capture cap)
        {
            Mat m = new Mat();
            cap.Retrieve(m);
            return hsv_frame = m.ToImage<Hsv, byte>();
        }
        public Image<Bgr, byte> GenerateCameraFrame_bgr(Capture cap)
        {
            Mat m = new Mat();
            cap.Retrieve(m);
            return bgr_frame = m.ToImage<Bgr, byte>();
        }
        public Image<Gray, byte> ToGray(Image<Bgr, byte> src)
        {
            this.bgr_frame = src;
            return this.imgGray = src.Convert<Gray, byte>();
        }
        public Image<Gray, byte> ToGray(Image<Hsv, byte> src)
        {
            this.hsv_frame = src;
            return this.imgGray = src.Convert<Gray, byte>();
        }
        public Image<Gray, byte> SkinFilter(Image<Bgr, byte> src,double[] color)
        {
            imgSkin = src.InRange(new Bgr(color[0], color[1], color[2]), new Bgr(color[3], color[4], color[5]));
            return this.imgSkin;
        }
        public Image<Gray, byte> SkinFilter(Image<Hsv, byte> src, double[] color)
        {
            imgSkin = src.InRange(new Hsv(color[0], color[1], color[2]), new Hsv(color[3], color[4], color[5]));
            return this.imgSkin;
        }
        public Image<Gray, byte> ToBinary(Image<Gray, byte> src,
                                          int thres_min = 50, int thres_max = 255,
                                          int dilate_t = 3, int erode_t = 3)
        {
            Image<Gray, byte> temp = src.Clone();
            return this.imgBinary = temp.InRange(new Gray(thres_min), new Gray(thres_max)).Dilate(dilate_t).Erode(erode_t);
        }

        /// contour / hull
        public VectorOfVectorOfPoint FindContours(Image<Gray, byte> src)
        {
            Image<Gray, byte> temp = src.Clone();

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hier = new Mat();
            CvInvoke.FindContours(temp, contours, hier,
                                  Emgu.CV.CvEnum.RetrType.External,
                                  Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxTc89Kcos);
            return contours;
        }
        public VectorOfInt FindConvexHull(VectorOfPoint contour)
        {
            VectorOfInt hull = new VectorOfInt();
            CvInvoke.ConvexHull(contour, hull, false, false);
            return hull;
        }
        public void DrawConvexHull(Image<Bgr, byte> src, VectorOfPoint contour, VectorOfInt hull)
        {
            /* contour = contours[i]
             * MessageBox.Show("con size:"   + contours.Size    + "\n" +    // 幾組封閉輪廓
             *                "con[i] size:"+ contours[i].Size + "\n" +    // 第i組輪廓點數量
             *                "hull size:"  + hull.Size        + "\n" +    // 第i組凸包點數量
             *                "hull[0]:"    + hull[0]          + "\n" +    // 第i組第0個凸包點索引
             *                "con[i][hull[0]]:" + contours[i][hull[0]].X ); // 第i組含有凸包的封閉輪廓 中第0個凸包點的索引 的X值
             */
            for (int j = 0; j < hull.Size; j++)
            {
                if (j != hull.Size - 1)
                    CvInvoke.Line(src, contour[hull[j]], contour[hull[j + 1]], new MCvScalar(255, 0, 0), 2);
                else
                    CvInvoke.Line(src, contour[hull[j]], contour[hull[0]], new MCvScalar(255, 0, 0), 2);
            }
        }

        /// circle
        public Point FindShapeInnerCircle(VectorOfPoint contour)
        {
            MCvMoments m = CvInvoke.Moments(contour);
            Point shape_center = new Point((int)(m.M10 / m.M00), (int)(m.M01 / m.M00));
            //CvInvoke.Circle(frame, shape_center, 30, new MCvScalar(0, 0, 1), -1);
            return shape_center;
        }
        public Point[] FindInnerCircle(VectorOfPoint contour, Rectangle image_RectRange)
        {
            Point[] inner_circle_data;
            double dist = 0;
            double maxdist = 0;
            Point center = new Point();
            for (double i = image_RectRange.Left; i <= image_RectRange.Right; i += 5)
            {
                for (double j = image_RectRange.Top; j < image_RectRange.Bottom; j += 5)
                {
                    dist = CvInvoke.PointPolygonTest(contour, new PointF((float)i, (float)j), true);
                    if (dist >= maxdist)
                    {
                        maxdist = dist;
                        center = new Point((int)i, (int)j);
                    }
                }
            }
            if (maxdist < 0) maxdist = -maxdist;
            inner_circle_data = new Point[] { center, new Point((int)maxdist, 0) };

            return inner_circle_data;  // [0] 圓心, [1] 半徑
        }
        public List<Point[]> FindInnerCircle2(VectorOfPoint contour, Rectangle image_RectRange)
        {
            List<Point[]> inner_circle_data = new List<Point[]>();
            double dist = 0;
            double maxdist = 0;
            Point center = new Point();
            for (double i = image_RectRange.Left; i <= image_RectRange.Right; i += 5)
            {
                for (double j = image_RectRange.Top; j < image_RectRange.Bottom; j += 5)
                {
                    dist = CvInvoke.PointPolygonTest(contour, new PointF((float)i, (float)j), true);
                    if ((dist > 20))
                    {
                        inner_circle_data.Add(new Point[] { new Point((int)i, (int)j), new Point((int)dist, 0) });
                    }
                    if (dist >= maxdist)
                    {
                        maxdist = dist;
                        center = new Point((int)i, (int)j);
                    }
                }
            }
            //if (maxdist < 0) maxdist = -maxdist;
            inner_circle_data.Add(new Point[] { center, new Point((int)maxdist, 0) });

            return inner_circle_data;  // [0] 圓心, [1] 半徑
        }
        public List<Dictionary<int, Point>> FindPointAtWhichSideOfPalm(VectorOfPoint palm_contour, VectorOfPoint contour, int n)
        {
            // n隨圖片大小向上調整
            List<Dictionary<int, Point>> PointData = new List<Dictionary<int, Point>>();
            Dictionary<int, Point> PointOutsideOfPalm = new Dictionary<int, Point>();
            Dictionary<int, Point> PointInsideOfPalm = new Dictionary<int, Point>();
            double dist = 0;
            bool isFirstOutside = false;
            int first = 0;
            int c = 0;   //在同一邊連續多少的點

            int i = 0;   //索引值
            for (; i < contour.Size; i++)
            {

                dist = CvInvoke.PointPolygonTest(palm_contour, contour[i], true);
                // 屬於外面
                if (dist < 0)
                {
                    if (i == 0 || isFirstOutside == true)
                    {
                        first++;
                        isFirstOutside = true;
                    }
                    PointOutsideOfPalm.Add(i, contour[i]);
                    c++;
                }
                // 屬於裡面或在邊界上
                else
                {
                    isFirstOutside = false;
                    if (c != 0 && c <= n)
                    {
                        for (int j = i - c; j <= i; j++)
                        {
                            PointOutsideOfPalm.Remove(j);
                            PointInsideOfPalm.Add(j, contour[j]);
                        }
                        c = 0;
                    }
                    else
                    {
                        PointInsideOfPalm.Add(i, contour[i]);
                        c = 0;
                    }
                }
            }

            // 末端少 
            if (c <= n)
            {
                //初端有值
                if (first > 0 && first <= n)
                {
                    // 末+初 少 -> 末端換掉
                    if ((c + first) <= n)
                    {
                        for (int j = i - c; j < contour.Size; j++)
                        {
                            PointOutsideOfPalm.Remove(j);
                            PointInsideOfPalm.Add(j, contour[j]);
                        }

                    }
                    // 末+初 多 -> 初端換掉
                    else if ((c + first) > n)
                    {
                        for (int j = 0; j < first; j++)
                        {
                            PointInsideOfPalm.Remove(j);
                            PointOutsideOfPalm.Add(j, contour[j]);
                        }
                    }
                }
            }
            // 末端多 初端有值
            else if (c > n && (first > 0 && first <= n))
            {
                for (int j = 0; j < first; j++)
                {
                    PointInsideOfPalm.Remove(j);
                    PointOutsideOfPalm.Add(j, contour[j]);
                }
            }


            PointData.Add(PointInsideOfPalm);
            PointData.Add(PointOutsideOfPalm);
            return PointData;
        }

        /// Angle / Tip
        //X
        public Dictionary<int, Point[]> FindTip(Dictionary<int, Point[]> angle_dict, Point center, double angle, int k)
        {
            int AngleDictCount = angle_dict.Count;
            bool isPluse = false;
            int count = 0;
            for (int i = 0; i < AngleDictCount; i++)
            {
                if (angle_dict.Values.ElementAt(i)[3].X <= angle)
                    if (i < k)
                    { }
                if (angle_dict.Values.ElementAt(i)[3].X <= angle_dict.Values.ElementAt(i - k)[3].X &&
                    angle_dict.Values.ElementAt(i)[3].X >= angle_dict.Values.ElementAt(i + k)[3].X)
                {
                    // 若 終點至手心距離 < 指尖至手心距離
                    if (i != 0 && isPluse == false)
                    {
                        //if (Dist_pow2(angle_dict.Values.ElementAt(i)[1], center) <= Dist_pow2(angle_dict.Values.ElementAt(i)[0], center))
                        {
                            count++;
                            isPluse = true;
                        }
                    }
                }
            }
            return null;
        }
        public Dictionary<int, Point> FindPalmContourPoint(VectorOfPoint contour, Point center, double radius, int gap_angle = 5)
        {
            Dictionary<int, Point> palm_contour = new Dictionary<int, Point>();
            int[] contour_count = new int[contour.Size]; // 紀錄該輪廓點被選取的次數
            double x = -1, y = -1;
            double dist = -1;
            double min_dist = -1;
            int min_dist_index = -1;

            for (int i = 0; i < 360; i += gap_angle)
            {
                // 圓周方程式
                x = center.X + radius * Math.Cos(i * Math.PI / 180.0);
                y = center.Y + radius * Math.Sin(i * Math.PI / 180.0);
                // 圓周方程式逆時針找最近輪廓點 找360/gap_angle個
                for (int j = 0; j < contour.Size; j++)
                {
                    dist = Dist_pow2(x, y, contour[j].X, contour[j].Y);
                    if (j == 0) //先預設第一個點為最靠近圓周的點
                    {
                        min_dist = dist;
                        min_dist_index = j;
                    }
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        min_dist_index = j;
                    }
                }

                // 儲存最近的輪廓點 索引為上面迴圈找到的 min_dist_index
                if (palm_contour.ContainsKey(min_dist_index) == false) // 如果不存在該索引就新增此輪廓點 (若該最近輪廓點又被選到一次就略過)
                {
                    // ******** 是否有確定有按照索引順序排序 *******
                    palm_contour.Add(min_dist_index, contour[min_dist_index]);
                }             
            }

            return palm_contour;
        }
        public List<Point> FindWristLine(VectorOfPoint contour)
        {
            List<Point> wrist = new List<Point>();
            double max_dist = 0;
            int max_index = 0;
            int max_y = 0;
            double second_dist = 0;
            int second_index = 0;
            int second_y = 0;
            double dist = 0;

            // 0 到 最後前一項
            for (int i = 0; i < contour.Size - 1; i++)
            {
                dist = Dist_pow2(contour[i], contour[i + 1]);
                if (dist > max_dist)
                {
                    second_dist = max_dist;
                    second_index = max_index;
                    second_y = max_y;
                    max_dist = dist;
                    max_index = i;
                    max_y = (contour[i].Y + contour[i + 1].Y) / 2;
                }
                else if (dist <= max_dist && dist > second_dist)
                {
                    second_dist = dist;
                    second_index = i;
                    second_y = (contour[i].Y + contour[i + 1].Y) / 2;
                }
            }

            // 最後一項
            dist = Dist_pow2(contour[contour.Size - 1], contour[0]);
            if (dist > max_dist)
            {
                second_dist = max_dist;
                second_index = max_index;
                second_y = max_y;
                max_dist = dist;
                max_index = contour.Size - 1;
                max_y = (contour[contour.Size - 1].Y + contour[0].Y) / 2;
            }
            else if (dist <= max_dist && dist > second_dist)
            {
                second_dist = dist;
                second_index = contour.Size - 1;
                second_y = (contour[contour.Size - 1].Y + contour[0].Y) / 2;
            }

            // 選取兩個當中偏下方的
            if (second_y > max_y)
            {
                max_dist = second_dist;
                max_index = second_index;
                max_y = second_y;
            }

            if (max_index == contour.Size - 1)
            {
                wrist.Add(contour[max_index]);
                wrist.Add(contour[0]);
            }
            else
            {
                wrist.Add(contour[max_index]);
                wrist.Add(contour[max_index + 1]);
            }


            wrist.Add(new Point((wrist[0].X + wrist[1].X) / 2, (wrist[0].Y + wrist[1].Y) / 2));

            return wrist; // 手腕點1,手腕點2,手腕中點
        }
        public List<Point> FindWristLine2(VectorOfPoint contour,double inner_radius) //加速判斷版
        {
            List<Point> wrist = new List<Point>();
            int dist_bound = (int)(0.8 * inner_radius); // 若候選手腕線>0.8*內接圓半徑
            double max_dist = 0;
            int max_index = 0;
            int max_y = 0;
            double second_dist = 0;
            int second_index = 0;
            int second_y = 0;
            double dist = 0;

            // 0 到 最後前一項
            for (int i = 0; i < contour.Size - 1; i++)
            {
                dist = Dist_pow2(contour[i], contour[i + 1]);
                if (dist > dist_bound) // 若候選手腕線>0.8*內接圓半徑
                {
                    if (dist > max_dist)
                    {
                        second_dist = max_dist;
                        second_index = max_index;
                        second_y = max_y;
                        max_dist = dist;
                        max_index = i;
                        max_y = (contour[i].Y + contour[i + 1].Y) / 2;
                    }
                    else if (dist <= max_dist && dist > second_dist)
                    {
                        second_dist = dist;
                        second_index = i;
                        second_y = (contour[i].Y + contour[i + 1].Y) / 2;
                    }
                }
            }

            // 最後一項
            dist = Dist_pow2(contour[contour.Size - 1], contour[0]);
            if (dist > dist_bound)
            {
                if (dist > max_dist)
                {
                    second_dist = max_dist;
                    second_index = max_index;
                    second_y = max_y;
                    max_dist = dist;
                    max_index = contour.Size - 1;
                    max_y = (contour[contour.Size - 1].Y + contour[0].Y) / 2;
                }
                else if (dist <= max_dist && dist > second_dist)
                {
                    second_dist = dist;
                    second_index = contour.Size - 1;
                    second_y = (contour[contour.Size - 1].Y + contour[0].Y) / 2;
                }
            }

            // 選取兩個當中偏下方的
            if (second_y > max_y)
            {
                max_dist = second_dist;
                max_index = second_index;
                max_y = second_y;
            }

            if (max_index == contour.Size - 1)
            {
                wrist.Add(contour[max_index]);
                wrist.Add(contour[0]);
            }
            else
            {
                wrist.Add(contour[max_index]);
                wrist.Add(contour[max_index + 1]);
            }


            wrist.Add(new Point((wrist[0].X + wrist[1].X) / 2, (wrist[0].Y + wrist[1].Y) / 2));

            return wrist; // 手腕點1,手腕點2,手腕中點
        }
        public Dictionary<int, Point> DeleteUnderWristLine(Point w1, Point w2, VectorOfPoint contours)
        {
            // 只取手腕線斜上方的手輪廓 點帶入手腕線方程小於0者
            Dictionary<int, Point> hand = new Dictionary<int, Point>();
            double m = (double)(w1.Y - w2.Y) / (double)(w1.X - w2.X);
            for (int i = 0; i < contours.Size; i++)
            {
                if (m * (contours[i].X - w1.X) + w1.Y - contours[i].Y >= 0.0)
                {
                    hand.Add(i, contours[i]);
                }
            }
            return hand;
        }

        public List<List<Point>> DividePalmFinger(VectorOfPoint contour, VectorOfPoint palm_con, int k = 5)
        {
            /// *******尚未判斷該點是否在輪廓內還是外還是線上 
            List<Point> finger = new List<Point>();
            List<List<Point>> finger_candidate_group = new List<List<Point>>();
            double dist = 0;


            /// 先找到最近的低谷 方便之後代碼簡潔
            int first_valley_index = 0;
            for (int i = 0; i < contour.Size; i++)
            {
                dist = CvInvoke.PointPolygonTest(palm_con, contour[i], true);
                if (dist == 0) // 若輪廓點在手心輪廓上時
                {
                    first_valley_index = i;
                    break;
                }
            }

            /// 開始從低谷找~~
            bool start = false;
            int begin_index = 0;
            int count = 0;
            // 第一個低谷點 至 輪廓最後一點
            for (int i = first_valley_index; i < contour.Size; i++)
            {
                dist = CvInvoke.PointPolygonTest(palm_con, contour[i], true);
                if (dist < 0) // 若輪廓點在手心輪廓外時
                {
                    if (start == false)
                    {
                        if (i != 0)
                            begin_index = i - 1;    // 紀錄起始點
                        else
                            begin_index = contour.Size - 1;
                        start = true;
                    }

                    finger.Add(contour[i]);
                    count++;
                }
                else
                {
                    if (start == true)
                    {
                        if (count > k)            //至少k個點組成的點集合才納入
                        {
                            finger.Insert(0, contour[begin_index]); // 新增起始邊界點
                            finger.Insert(0, new Point((contour[i].X + contour[begin_index].X) / 2,
                                                (contour[i].Y + contour[begin_index].Y) / 2));
                            finger.Insert(0, contour[i]);            // 新增末邊界點
                            finger_candidate_group.Add(finger); // 新增一組手指輪廓
                            i--;
                        }

                        finger = new List<Point>();
                        start = false;
                        count = 0;
                    }
                }
            }
            // 第一個點 至 第一個低谷點
            for (int i = 0; i <= first_valley_index; i++)
            {
                dist = CvInvoke.PointPolygonTest(palm_con, contour[i], true);
                if (dist < 0)
                {
                    if (start == false)
                    {
                        if (i != 0)
                            begin_index = i - 1;    // 紀錄起始點
                        else
                            begin_index = contour.Size - 1;
                        start = true;
                    }

                    finger.Add(contour[i]);
                    count++;
                }
                else
                {
                    if (start == true)
                    {
                        if (count > k)    //至少k個點組成的點集合才納入
                        {
                            finger.Insert(0, contour[begin_index]); // 新增起始邊界點
                            finger.Insert(0, new Point((contour[i].X + contour[begin_index].X) / 2,
                                                (contour[i].Y + contour[begin_index].Y) / 2));
                            finger.Insert(0, contour[i]);            // 新增末邊界點
                            finger_candidate_group.Add(finger); // 新增一組手指輪廓
                            i--;
                        }

                        finger = new List<Point>();
                        start = false;
                        count = 0;
                    }
                }
            }
            return finger_candidate_group;
        }
        public List<Point> FindFingerTip(List<Point> finger, int k = 5)
        {

            Point finger_base = finger[1];
            double start_dist = 0, this_dist = 0, end_dist = 0;
            Point finger_tip = new Point();
            int count = 0;
            List<Point> finger_tip_group = new List<Point>();

            for (int i = 2 + k; i < finger.Count - k; i++)
            {
                start_dist = Dist(finger_base, finger[i - k]);
                this_dist = Dist(finger_base, finger[i]);
                end_dist = Dist(finger_base, finger[i + k]);
                if (this_dist > start_dist && this_dist > end_dist)
                {
                    finger_tip = finger[i];
                    count++;
                }
                if (count > 0 && (this_dist < start_dist && this_dist < end_dist))
                {
                    finger_tip_group.Add(finger_tip);
                    count = 0;
                }
            }
            finger_tip_group.Add(finger_tip);

            return finger_tip_group;
        }
        public List<List<Point>> FindFingerTip2(List<List<Point>> fingers, int k = 5)
        {
            // fingers[][0]: 手指群 最後一點 
            // fingers[][1]: 手指群 指底中點
            // fingers[][2]: 手指群 第一個點
            // ...

            List<Point> finger = new List<Point>();
            List<List<Point>> finger_data = new List<List<Point>>();

            for (int i = 0; i < fingers.Count; i++)
            {
                Point finger_base = fingers[i][1];
                double start_dist = 0, this_dist = 0, end_dist = 0;
                Point finger_tip = new Point(-1, -1);
                int finger_tip_index = -1;
                bool isTip = false;
                int tip_count = 0; //數共有幾根手指頭

                int begin_index = 2;
                for (int j = 2 + k; j < fingers[i].Count - k; j++)
                {
                    start_dist = Dist(finger_base, fingers[i][j - k]);
                    this_dist = Dist(finger_base, fingers[i][j]);
                    end_dist = Dist(finger_base, fingers[i][j + k]);

                    // 找是否為指尖點
                    if (this_dist > start_dist && this_dist > end_dist)
                    {                   
                        finger_tip = fingers[i][j];
                        finger_tip_index = j;
                        isTip = true;
                    }

                    // 若找到指縫點
                    if (isTip == true && (this_dist < start_dist && this_dist < end_dist))
                    {
                        // 新增一根手指
                        if (cosine(finger_tip, fingers[i][j], fingers[i][begin_index]) > 0.5) // 指尖夾角小於60度
                        {
                            finger.Add(fingers[i][j]);                                              // 新增最末點0
                            finger.Add(new Point((fingers[i][j].X + fingers[i][begin_index].X) / 2, // 新增指底點1
                                                 (fingers[i][j].Y + fingers[i][begin_index].Y) / 2));
                            finger.Add(fingers[i][begin_index]);                                    // 新增起始點2                      
                            finger.Add(fingers[i][(begin_index * 2 + finger_tip_index * 1) / 3]);    // 新增第3點
                            finger.Add(fingers[i][(begin_index * 1 + finger_tip_index * 2) / 3]);    // 新增第4點
                            finger.Add(finger_tip);                                                 // 新增指尖點 5
                            finger.Add(fingers[i][(finger_tip_index * 2 + j * 1) / 3]);             // 新增第6點
                            finger.Add(fingers[i][(finger_tip_index * 1 + j * 2) / 3]);             // 新增第7點

                            finger_data.Add(finger);         // 新增一根手指
                            finger = new List<Point>();

                            tip_count++;
                        }
                            begin_index = j;
                            isTip = false;
                    }
                    else if (j == fingers[i].Count - k - 1 && finger_tip.X != -1)
                    {
                        // 新增一根手指
                        if (cosine(finger_tip, fingers[i][0], fingers[i][begin_index]) > 0.5) // 指尖夾角小於60度
                        {
                            finger.Add(fingers[i][0]);                                            // 新增最末點0
                            finger.Add(new Point((fingers[i][0].X + fingers[i][begin_index].X) / 2, // 新增指底點1
                                                 (fingers[i][0].Y + fingers[i][begin_index].Y) / 2));
                            finger.Add(fingers[i][begin_index]);                                  // 新增起始點2
                            finger.Add(fingers[i][(begin_index * 2 + finger_tip_index * 1) / 3]); // 新增第3點
                            finger.Add(fingers[i][(begin_index * 1 + finger_tip_index * 2) / 3]); // 新增第4點
                            finger.Add(finger_tip);                                               // 新增指尖點
                            finger.Add(fingers[i][(finger_tip_index * 2 + j * 1) / 3]);           // 新增第6點
                            finger.Add(fingers[i][(finger_tip_index * 1 + j * 2) / 3]);           // 新增第7點

                            finger_data.Add(finger);             // 新增一根手指
                            finger = new List<Point>();
                        }
                    }
                }
            }

            return finger_data;
        }
        public bool haveThumbs(Point wrist, Point center, Point finger_base)
        {
            bool isThumb = false;
            double rad = cosine(finger_base, wrist, center); // 計算指底至手心與指底至手腕中點的夾角
            double k = 0; //指底與手心手腕中點的相對位置計算 偏下方的為拇指

            if (wrist.Y - center.Y == 0)
            {
                double _x = 1.5 * center.X - 0.5 * wrist.X;
                if (wrist.X >= center.X)
                {
                    if (finger_base.X > _x)
                        k = -1;
                    else
                        k = 1;
                }
                else
                {
                    if (finger_base.X < _x)
                        k = -1;
                    else
                        k = 1;
                }

            }
            else
            {
                double m2 = (center.X - wrist.X) / (wrist.Y - center.Y);
                double _x = 1.5 * center.X - 0.5 * wrist.X;
                double _y = 1.5 * center.Y - 0.5 * wrist.Y;
                k = m2 * (finger_base.X - _x) - (finger_base.Y - _y);
            }


            if (rad < 0.86 && k<=0)
                isThumb = true;
            else
                isThumb = false;

            return isThumb;

        }
        public List<int> fingers_index(List<Point> wrist, Point center, List<List<Point>> fingers)
        {
            List<int> finger_index = new List<int>();
            int ThumbSide = 0; // 0:沒有大拇指 1:大拇指在左邊 2:大拇指在右邊

            int wrist_vector_x = wrist[0].X - wrist[1].X; // 若輪廓找尋是 逆/(順)時針 則反過來相減
            int wrist_vector_y = wrist[0].Y - wrist[1].Y;
            double m = (double)wrist_vector_y / (double)wrist_vector_x;

            double min_angle = 360;
            int min_index = -1, thumb_index = -1;

            // 先找指底離手腕線最近的
            double min_dist = 100000, dist = 0;
            for (int i = 0; i < fingers.Count; i++)
            {
                dist = Math.Abs(m * (fingers[i][5].X - wrist[1].X) - fingers[i][5].Y + wrist[1].Y); // 有些是常數所以省略掉多餘計算
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_index = i;
                }
            }

            // 檢測指底至手心中心 與 手腕線 夾角 是否吻合大拇指條件
            int temp_thumb_vector_x = fingers[min_index][5].X - center.X;
            int temp_thumb_vector_y = fingers[min_index][5].Y - center.Y;
            double temp_thumb_angle = Math.Acos(cosine(new Point(0, 0),
                                  new Point(wrist_vector_x, wrist_vector_y),
                                  new Point(temp_thumb_vector_x, temp_thumb_vector_y))) * 180.0 / Math.PI;


            // 若夾角大於90 可能是拇指的會在左手邊
            if (temp_thumb_angle > 90.0)
            {
                temp_thumb_angle = 180.0 - temp_thumb_angle;
                if (temp_thumb_angle < 35.0)// 若夾角小於 35 度  條參數                                       
                {
                    thumb_index = min_index;
                    finger_index.Add(thumb_index);
                    ThumbSide = 1;
                }
                else
                {
                    ThumbSide = 0;
                }
            }
            else
            {
                if (temp_thumb_angle < 35.0)     // 若夾角小於 35 度 調參數
                {
                    thumb_index = min_index;
                    finger_index.Add(thumb_index);
                    ThumbSide = 2;
                }
                else
                {
                    ThumbSide = 0;
                }
            }

            int min_ii = -1;
            for (int i = 0; i < fingers.Count; i++)
            {
                if (i != thumb_index)
                {
                    int finger_vector_x = fingers[i][5].X - center.X;
                    int finger_vector_y = fingers[i][5].Y - center.Y;
                    double finger_angle = cosine(new Point(0, 0),
                          new Point(wrist_vector_x, wrist_vector_y),
                          fingers[i][fingers[i].Count - 2]) * 180.0 / Math.PI;  // 指底至手心中心 與 手腕線 夾角
                    if (finger_angle < min_angle)
                    {
                        min_angle = finger_angle;
                        min_ii = i;
                    }
                }
            }


            // 大拇指在右邊 待改
            if (ThumbSide == 2)
            {
                MessageBox.Show("Right");
                for (int i = thumb_index; i < fingers.Count; i++)
                {
                    if (i != thumb_index)
                        finger_index.Add(i);
                }
                if (thumb_index > 0)
                {
                    for (int i = 0; i < thumb_index; i++)
                    {
                        finger_index.Add(i);
                    }
                }
            }
            // 大拇指在左邊 已正確
            else if (ThumbSide == 1)
            {
                //MessageBox.Show("Left");
                if (thumb_index > 0)
                {
                    for (int i = thumb_index - 1; i >= 0; i--)
                    {
                        finger_index.Add(i);
                    }
                }

                for (int i = fingers.Count - 1; i >= thumb_index; i--)
                {
                    if (i != thumb_index)
                        finger_index.Add(i);
                }

            }
            // 沒有大拇指
            else
            {
                for (int i = min_ii; i < fingers.Count; i++)
                {
                    finger_index.Add(i);
                }
                if (min_ii != 0)
                {
                    for (int i = 0; i < min_ii; i++)
                    {
                        finger_index.Add(i);
                    }
                }
            }

            return finger_index;
        }
    }
}

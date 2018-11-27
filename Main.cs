using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;

using System.Windows.Forms.DataVisualization.Charting;
using System.IO.Ports;
using System.Diagnostics;



namespace HandGesture_Recognition_Project
{
    public partial class Start_label : Form
    {
        ////SerialPort port = new SerialPort("COM4", 115200, Parity.None, 8, StopBits.One);
        ////Byte[] port_data = new Byte[1];
        bool start_img_programming = false;
        int port_msg = 0;

        // img
        Capture cap; // 640 * 480 ( cap.Width , cap.Height )
        double[] Ratio = new double[2] { 1,1};
        Recognition_v1 my = new Recognition_v1();
        Image<Hsv, byte> hsv_frame;
        Image<Bgr, byte> bgr_frame;
        Image<Gray, byte> gray;
        Image<Gray, byte> input;

        //event
        int[] variable = { 50,255,20,20 }; // thres,thres_link,dilate_t,erode_t
        int k_curvature = 1, angle = 120, angle_variance = 20;
        int rect_width = 0, rect_height = 0;
        //double[] hsv_array = { 34,0,34,255,135,255};
        double[] hsv_array = { 0, 0, 0, 173, 173, 255 };

        bool drawing = false;
        bool erase = false;

        /// <summary>
        ///  初始化
        /// </summary>
        public Start_label()
        {
            InitializeComponent();
            Hmin_trackBar.ValueChanged += new EventHandler(Bmin_trackBar_ValueChanged);
            //Hmin_trackBar.ValueChanged += new EventHandler(Bmax_trackBar_ValueChanged);
            //if (port.IsOpen == false)
            //    port.Open();
        }

        Point start = new Point();
        Rectangle rect = new Rectangle();
        private void picturebox_Main_MouseDown(object sender, MouseEventArgs e)
        {
            start = e.Location;
        }
        private void picturebox_Main_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;
            Point temp_end = e.Location;
            rect.Location = new Point((int)(Math.Min(start.X, temp_end.X) * Ratio[0]), (int)(Math.Min(start.Y, temp_end.Y) * Ratio[1]));
            rect.Size = new Size((int)(Math.Abs(start.X - temp_end.X) * Ratio[0]), (int)(Math.Abs(start.Y - temp_end.Y) * Ratio[1]));
        }
        private void picturebox_Main_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;
            CvInvoke.Rectangle(bgr_frame, rect, new MCvScalar(0, 255, 255));
            start_img_programming = true;
            //picturebox_Main.Invalidate();
            //work(k_curvature, angle, angle_variance);
        }

        // 一個鏡頭擷取圖片的事件
        private void Cap_ImageGrabbed(object sender, EventArgs e)
        {
            process(cap);
        }

        private void process(Capture cap)
        {
            //Stopwatch stopwatch = new Stopwatch();
            //stopwatch.Start();
            cap.FlipHorizontal = true;
            hsv_frame = my.GenerateCameraFrame_hsv(cap);
            bgr_frame = my.GenerateCameraFrame_bgr(cap);

            if (start_img_programming == true)
            {
                if (rect != null)
                {
                    hsv_frame.ROI = rect;
                    bgr_frame.ROI = rect;
                    rect_width = rect.Width;
                    rect_height = rect.Height;
                }
                if (Skin_checkBox.Checked == true)
                {
                    input = my.SkinFilter(hsv_frame, hsv_array);
                }
                else
                {
                    gray = my.ToGray(hsv_frame);
                    input = my.ToBinary(gray, variable[0], variable[1], variable[2], variable[3]);
                }

                my.Gesture_v1(bgr_frame, input, k_curvature, angle);

                bgr_frame.ROI = new Rectangle(new Point(0, 0), new Size(cap.Width, cap.Height));

                port_msg = draw_panel();

                //stopwatch.Stop();
                //TimeSpan ts = stopwatch.Elapsed;
                //CvInvoke.PutText(bgr_frame, "" + ts.Milliseconds ,new Point(10, 30), Emgu.CV.CvEnum.FontFace.HersheyComplex, 1, new MCvScalar(0, 0, 255));
                //寫在下方的recognition_v1中了/CvInvoke.PutText(bgr_frame, "" + my.gesture, new Point(10, 60), Emgu.CV.CvEnum.FontFace.HersheyComplex, 1, new MCvScalar(0, 0, 255));
                picturebox_Binary.Image = input.Bitmap;
            }
            picturebox_Main.Image = bgr_frame.Bitmap;

            ////if (port.IsOpen == false)
            ////{
            ////    port.Open();
            ////}
            ////if (my.gesture >= 0 && my.gesture <= 5)
            ////{
            ////    port_data[0] = (byte)port_msg;
            ////    port.Write(port_data, 0, port_data.Length);
            ////    port_data = new Byte[1];
            ////}
        }

        List<Point> draw_timeline = new List<Point>(); // 儲存畫筆軌跡 時間線
        List<Point> trace_timeline = new List<Point>(); // 控制風扇角度
        bool trace_original = true;
        List<int> gesture_timeline = new List<int>();   // 儲存手勢類別 時間線
        int brightness = 0;

        private bool InRegion(Point p, int start_x, int start_y, int width, int height)
        {
            if ((p.X > start_x && p.X < start_x + width) && (p.Y > start_y && p.Y < start_y + height))
                return true;
            else
                return false;
        }
        private void draw_rect(int start_x,int start_y,int width,int height,MCvScalar color)
        {
            // 統一左上角為起始點 -> 右上 -> 右下 -> 左下
            CvInvoke.Line(bgr_frame, new Point(start_x , start_y), new Point(start_x + width, start_y), color, 5);
            CvInvoke.Line(bgr_frame, new Point(start_x + width, start_y), new Point(start_x + width, start_y + height), color, 5);
            CvInvoke.Line(bgr_frame, new Point(start_x + width, start_y + height), new Point(start_x , start_y + height), color, 5);
            CvInvoke.Line(bgr_frame, new Point(start_x, start_y + height), new Point(start_x, start_y), color, 5);
        }
        private void draw_filled_rect(int brightness)
        {
            draw_rect(rect.X + rect.Width - 40 - 20, rect.Y + 140, 40, 40, new MCvScalar(brightness, brightness, brightness));
            draw_rect(rect.X + rect.Width - 40 - 15, rect.Y + 145, 30, 30, new MCvScalar(brightness, brightness, brightness));
            draw_rect(rect.X + rect.Width - 40 - 10, rect.Y + 150, 20, 20, new MCvScalar(brightness, brightness, brightness));
            draw_rect(rect.X + rect.Width - 40 - 5, rect.Y + 155, 10, 10, new MCvScalar(brightness, brightness, brightness));
            draw_rect(rect.X + rect.Width - 40 - 3, rect.Y + 157, 6, 6, new MCvScalar(brightness, brightness, brightness));
            draw_rect(rect.X + rect.Width - 40 - 1, rect.Y + 159, 2, 2, new MCvScalar(brightness, brightness, brightness));
        }
        private int draw_panel()
        {
            int mode = 9;
            ////////////////////// 外框 ////////////////////////
            draw_rect(rect.X, rect.Y, rect.Width, rect.Height, new MCvScalar(0, 255, 255));

            ////////////////////// 建圖 /////////////////////
            Point offset_position = new Point(my.position.X + rect.X, my.position.Y + rect.Y);
            draw_rect(rect.X + rect.Width - 40 - 20, rect.Y + 20, 40, 40, new MCvScalar(0, 0, 255)); //畫筆按鈕
            draw_rect(rect.X + rect.Width - 40 - 20, rect.Y + 80, 40, 40, new MCvScalar(255, 255, 255)); //清除按鈕

            ///////////////////// 各手勢操作 ///////////////
            /// 手勢1
            if (my.gesture == 1)
            {
                /////// 若在按鈕區域內 ///////
                if (InRegion(offset_position, rect.X + rect.Width - 40 - 20, rect.Y + 20, 40, 40) == true)
                {
                    drawing = true;
                    erase = false;
                    mode = 9;
                }
                else if (InRegion(offset_position, rect.X + rect.Width - 40 - 20, rect.Y + 80, 40, 40) == true)
                {
                    drawing = false;
                    erase = true;
                    mode = 9;
                }

                ///// 若選擇畫線按鈕--畫線 ******/
                if (drawing == true)
                {
                    draw_timeline.Add(offset_position);
                    if (draw_timeline.Count == 30)
                    {
                        draw_timeline.RemoveAt(0);
                    }
                    for (int j = draw_timeline.Count - 1; j > 0; j--)
                    {
                        CvInvoke.Line(bgr_frame, draw_timeline[j], draw_timeline[j - 1], new MCvScalar(0, 0, 0), j);
                    }
                    mode = 9;
                }
                else ////// 若沒選擇畫線按鈕--風扇旋轉角度控制 ///////
                {
                    trace_timeline.Add(offset_position);
                    if (trace_original == true)
                    {
                        trace_original = false;
                    }
                    else
                    {
                        int del_x = trace_timeline[1].X - trace_timeline[0].X;
                        trace_timeline.RemoveAt(0);
                        if (del_x > 3)
                        {
                            mode = 7;
                        }
                        else if (del_x < -3)
                        {
                            mode = 8;
                        }
                        else
                        {
                            mode = 9;
                        }
                    }
                }

                ///// 若選擇清除按鈕--清除畫線 ////
                if (erase == true)
                {
                    draw_timeline = new List<Point>();
                    erase = false;
                    mode = 9;
                }                
            }
            /// 手勢5
            else if (my.gesture == 5) //////////////////////// 亮度調整顯示  //////////////////
            {
                if (brightness <= 245)
                {
                    brightness += 10;
                }
                //draw_filled_rect(brightness);
                mode = 5;
            }
            /// 手勢0
            else if (my.gesture == 0)
            {
                if (brightness >= 10)
                {
                    brightness -= 10;
                }
                //draw_filled_rect(brightness);
                mode = 0;
            }
            /// 手勢 2,3,4,其他
            else
            {
                //draw_filled_rect(brightness);
                mode = 9;
            }

            CvInvoke.PutText(bgr_frame, "" + mode, new Point(30 + +rect.X, 30 + rect.Y), Emgu.CV.CvEnum.FontFace.HersheyComplex, 1, new MCvScalar(0, 255, 0));

            return mode;
        }

        // mouse click
        //  一個Start鈕按下左鍵的事件
        private void Start_button_Click(object sender, EventArgs e)
        {
            Stop_label.Enabled = true;
            Start_button.Enabled = false;

            if (cap == null && Stop_label.Enabled == true)
            {
                cap = new Capture(0);
                Ratio[0] = (double)cap.Width / (double)picturebox_Main.Width;
                Ratio[1] = (double)cap.Height / (double)picturebox_Main.Height;
                MessageBox.Show(cap.Width + "," + cap.Height + "," + picturebox_Main.Width + "," + picturebox_Main.Height);
            }

            if (cap != null)
            {
                cap.ImageGrabbed += new EventHandler(Cap_ImageGrabbed); //發生cap_imagegrabbed(鏡頭擷取一張張影像)的事件 
                cap.Start();
                label_rect.Text = rect_width + " x " + rect_height;
            }
        }
        //  一個Stop鈕按下左鍵的事件
        private void Stop_label_Click(object sender, EventArgs e)
        {
            Start_button.Enabled = true;
            Stop_label.Enabled = false;
            cap.Stop();
        }
        //chart
        private void button_chart_Click(object sender, EventArgs e)
        {
            my_chart.Series.Clear();
            my_chart.ChartAreas[0].AxisX.Minimum = 0;
            my_chart.ChartAreas[0].AxisX.Maximum = my.InputContour.Size;
            my_chart.ChartAreas[0].AxisY.Minimum = 0;
            my_chart.ChartAreas[0].AxisY.Maximum = 300;

            Series In_Dist_Chart = new Series();
            In_Dist_Chart.Color = Color.DarkOrange;
            In_Dist_Chart.Font = new System.Drawing.Font("新細明體", 10); //設定字型
            In_Dist_Chart.ChartType = SeriesChartType.Point; //設定線條種類                                                                                                      

            foreach (var item in my.all_dist_dict[0])
            {
                In_Dist_Chart.Points.AddXY(item.Key, item.Value[1]  /* my.all_dist_dict[0].Values.ElementAt(0)[1] * 120*/);
            }

            Series Out_Dist_Chart = new Series();
            Out_Dist_Chart.Color = Color.DarkBlue;
            Out_Dist_Chart.Font = new System.Drawing.Font("新細明體", 10); //設定字型
            Out_Dist_Chart.ChartType = SeriesChartType.Point; //設定線條種類   

            foreach (var item in my.all_dist_dict[1])
            {
                Out_Dist_Chart.Points.AddXY(item.Key, item.Value[1]  /* my.all_dist_dict[0].Values.ElementAt(0)[1] * 120*/);
            }

            //my_chart.Series.Add(Angle_Chart);
            //my_chart.Series.Add(Dist_Chart);
            //my_chart.Series.Add(Partial_Max_Dist_Chart);
            my_chart.Series.Add(In_Dist_Chart);
            my_chart.Series.Add(Out_Dist_Chart);
        }

        // checkBox change
        private void Skin_checkBox_CheckedChanged(object sender, EventArgs e)
        {
            if (Skin_checkBox.Checked == true)
            {
                GrayScale_checkBox.Checked = false;
                thres_trackBar.Enabled = false;
                thres_link_trackBar.Enabled = false;
                Hmin_trackBar.Enabled = true;
                Hmax_trackBar.Enabled = true;
                Smin_trackBar.Enabled = true;
                Smax_trackBar.Enabled = true;
                Vmin_trackBar.Enabled = true;
                Vmax_trackBar.Enabled = true;
            }
            else
            {
                GrayScale_checkBox.Checked = true;
            }
        }
        private void GrayScale_checkBox_CheckedChanged(object sender, EventArgs e)
        {
            if (GrayScale_checkBox.Checked == true)
            {
                Skin_checkBox.Checked = false;
                Hmin_trackBar.Enabled = false;
                Hmax_trackBar.Enabled = false;
                Smin_trackBar.Enabled = false;
                Smax_trackBar.Enabled = false;
                Vmin_trackBar.Enabled = false;
                Vmax_trackBar.Enabled = false;
                thres_trackBar.Enabled = true;
                thres_link_trackBar.Enabled = true;
            }
            else
            {
                Skin_checkBox.Checked = true;
            }
        }

        // value change

        private void Bmin_trackBar_ValueChanged(object sender, EventArgs e)
        {
            hsv_array[0] = Hmin_trackBar.Value;
            Hmin_value_label.Text = "Hmin: " + Hmin_trackBar.Value;
        }
        private void Gmin_trackBar_ValueChanged(object sender, EventArgs e)
        {
            hsv_array[1] = Smin_trackBar.Value;
            Smin_value_label.Text = "Smin: " + Smin_trackBar.Value;
        }
        private void Rmin_trackBar_ValueChanged(object sender, EventArgs e)
        {
            hsv_array[2] = Vmin_trackBar.Value;
            Vmin_value_label.Text = "Vmin: " + Vmin_trackBar.Value;
        }
        private void Bmax_trackBar_ValueChanged(object sender, EventArgs e)
        {
            hsv_array[3] = Hmax_trackBar.Value;
            Hmax_value_label.Text = "Hmax: " + Hmax_trackBar.Value;
        }
        private void Gmax_trackBar_ValueChanged(object sender, EventArgs e)
        {
            hsv_array[4] = Smax_trackBar.Value;
            Smax_value_label.Text = "Smax: " + Smax_trackBar.Value;
        }
        private void Rmax_trackBar_ValueChanged(object sender, EventArgs e)
        {
            hsv_array[5] = Vmax_trackBar.Value;
            Vmax_value_label.Text = "Vmax: " + Vmax_trackBar.Value;
        }

        private void K_Curvature_trackBar_ValueChanged(object sender, EventArgs e)
        {
            K_Curvature_label.Text = "K_curvature: " + K_Curvature_trackBar.Value;
            k_curvature = K_Curvature_trackBar.Value;
        }
        private void Angle_trackBar_ValueChanged(object sender, EventArgs e)
        {
            Angle_label.Text = "Angle: " + Angle_trackBar.Value;
            angle = Angle_trackBar.Value;
        }
        private void AngleVariance_trackBar_ValueChanged(object sender, EventArgs e)
        {
            AngleVariance_label.Text = "AngleVariance : " + AngleVariance_trackBar.Value;
            angle_variance = AngleVariance_trackBar.Value;
        }

        private void thres_trackBar_ValueChanged(object sender, EventArgs e)
        {
            thres_value_label.Text = "" + thres_trackBar.Value;
            variable[0] = thres_trackBar.Value;
        }
        private void thres_link_trackBar_ValueChanged(object sender, EventArgs e)
        {
            thres_link_value_label.Text = "" + thres_link_trackBar.Value;
            variable[1] = thres_link_trackBar.Value;
        }
        private void dilation_trackBar_ValueChanged(object sender, EventArgs e)
        {
            dilation_times_label.Text = "" + dilation_trackBar.Value;
            variable[2] = dilation_trackBar.Value;
        }
        private void erosion_trackbar_ValueChanged(object sender, EventArgs e)
        {
            erotion_times_label.Text = "" + erosion_trackbar.Value;
            variable[3] = erosion_trackbar.Value;

        }
       
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Drawing;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;

namespace HandGesture_Recognition_Project
{
    class My_Math
    {
        protected double cosine(Point defect, Point start, Point end) //回傳兩向量間夾角的餘弦值
        {
            Point a = new Point((start.X - defect.X), (start.Y - defect.Y));
            Point b = new Point((end.X - defect.X), (end.Y - defect.Y));
            double numerater = a.X * b.X + a.Y * b.Y;
            double dominater = Math.Sqrt(a.X * a.X + a.Y * a.Y) * Math.Sqrt(b.X * b.X + b.Y * b.Y);
            if (numerater / dominater >= 1)
                return 1;
            else if (numerater / dominater <= -1)
                return -1;
            else
                return numerater / dominater;
        }
        protected double h_angle(Point start, Point end)
        {
            double dy = end.Y - start.Y;
            double dx = end.X - end.X;
            double angle = 0;

            if (dx > 0)
            {
                if (dy > 0)
                {
                    angle = Math.Atan(dy / dx) * 180.0 / Math.PI;
                }
                else
                {
                    angle = 360.0 - Math.Atan(-dy / dx) * 180.0 / Math.PI;
                }
            }
            else
            {
                if (dy > 0)
                {
                    angle = 180.0 - Math.Atan(-dy / dx) * 180.0 / Math.PI;
                }
                else
                {
                    angle = 180.0 + Math.Atan(dy / dx) * 180.0 / Math.PI;
                }
            }

            return angle;
        } // 向量與正X軸的夾角
        protected int Avg(Dictionary<int, Point[]> dict, int nums)
        {
            int total = 0;
            foreach (var item in dict)
            {
                total += item.Value[3].X;
            }
            return total / nums;
        }
        protected double Dist_pow2(Point a, Point b)
        {
            return (Math.Pow(a.X - b.X, 2) + Math.Pow(a.Y - b.Y, 2));
        } //求兩點距離平方
        protected double Dist_pow2(double ax, double ay, int bx, int by)
        {
            return Math.Pow(ax - bx, 2) + Math.Pow(ay - by, 2);
        } //求兩點距離平方
        protected double Dist(Point a, Point b)
        {
            return Math.Sqrt(Math.Pow(a.X - b.X, 2) + Math.Pow(a.Y - b.Y, 2));
        } //求兩點距離
        protected double Slope(Point a, Point b)
        {
            return Math.Atan2(b.Y - a.Y, b.X - a.X);
        }
        //protected PointF UnitVector(PointF a)
        //{
        //    double len = Math.Sqrt(a.X * a.X + a.Y * a.Y);
        //    return new PointF((float)(a.X / len), (float)(a.Y / len));
        //}
        protected Point LineDirection(Point defect, Point a, Point b)
        {
            Point start_vector = new Point(a.X - defect.X, a.Y - defect.Y);
            Point end_vector = new Point(b.X - defect.X, b.Y - defect.Y);

            return new Point(start_vector.X + end_vector.X, start_vector.Y + end_vector.Y);
        }
    }
}

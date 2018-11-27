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
    class My_Group_Processing:My_img_processing
    {
        public VectorOfInt Group_ConvexHull(VectorOfPoint contour, bool DrawHull = true)
        {
            VectorOfInt hull;
            hull = FindConvexHull(contour);
            if (DrawHull == true)
                DrawConvexHull(bgr_frame, contour, hull);
            return hull;
        }
        //public Dictionary<int, Point[]> Group_ConvexDefect(VectorOfPoint contour, VectorOfInt hull, bool DrawDefect = true)
        //{
        //    Dictionary<int, Point[]> defect_dict = FindConvexDefect(contour, hull);
        //    if (DrawDefect == true)
        //        DrawConvexDefect(contour, defect_dict);
        //    return defect_dict;
        //}
    }
}

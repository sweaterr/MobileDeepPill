package msu.ece.xiaozeng.mpf3.classifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.io.IOException;
import java.util.Arrays;
import java.util.Vector;

import static org.opencv.core.Core.randn;

/**
 * Created by xiao on 12/2/16.
 */

public class OpenCVTool {
    public final static String TAG = "OpenCV_Xiao";
    private Context context;

    public OpenCVTool(Context context){
        this.context = context;
    }

    public Vector<Long> runCoarseLocalization(int runTimes){
        //Mat hsvImg.create(img.size(), CvType.CV_8U);
        Mat grayImg = new Mat(new Size(800,600),CvType.CV_8U);
        Mat dstImg = new Mat(new Size(800,600),CvType.CV_8U);
        long time_1 = System.currentTimeMillis();
        while(runTimes>0) {
            Imgproc.Canny(grayImg, dstImg, 1.0, 3.0, 3, false);
            Imgproc.dilate(grayImg, dstImg, new Mat(5,5,CvType.CV_8U), new Point(-1, -1), 1);
            Imgproc.erode(grayImg, dstImg, new Mat(5,5,CvType.CV_8U), new Point(-1, -1), 1);
            runTimes--;
        }
        long time_2 = System.currentTimeMillis();
        Vector<Long> result = new Vector<>();
        result.add(time_1);
        result.add(time_2);
        Log.i(TAG,"Coarse Localization :"+(time_2-time_1));
        return result;
    }

    public Vector<Long> runFinedGrainedLocalization(int runTimes){
        HOGDescriptor mHOGDescriptor = new HOGDescriptor();
        mHOGDescriptor.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());
        final MatOfRect foundLocations = new MatOfRect();
        final MatOfDouble foundWeights = new MatOfDouble();
        final MatOfPoint foundPoints = new MatOfPoint();
        Mat img = readImage2Mat();
        Mat resizeImage = new Mat();
        Size sz = new Size(256,256);
        Imgproc.resize( img, resizeImage, sz );
        long time_1 = System.currentTimeMillis();
        while(runTimes>0) {
            mHOGDescriptor.detect(resizeImage, foundPoints, foundWeights);
            runTimes--;
        }
        //mHOGDescriptor.detectMultiScale(img, foundLocations, foundWeights, 0.0, winStride, padding, 1.05, 2.0, false);
        long time_2 = System.currentTimeMillis();
        Log.i(TAG,"Fine-grained time :"+(time_2-time_1));
        Rect[] array = foundLocations.toArray();
        for (int j = 0; j < array.length; j++) {
            Rect rect = array[j];
            Log.i(TAG, "Height " + rect.height + ", Width " + rect.width);
        }
        Vector<Long> result = new Vector<>();
        result.add(time_1);
        result.add(time_2);
        return result;
    }

    public Vector<Long> runRetrieval(int runTimes){
        //Mat hsvImg.create(img.size(), CvType.CV_8U);
        float[] scores = new float[2000];
        Mat mat1 = new Mat(1, 328, CvType.CV_32F);
        Mat mat2 = new Mat(328, 2000, CvType.CV_32F);
        Mat mat3 = new Mat(1, 2000, CvType.CV_32F);
        randn(mat1, 0, 0.1);
        randn(mat2, 0, 0.1);
        long time_1 = System.currentTimeMillis();
        while(runTimes>0) {
            Core.gemm(mat1, mat2, 1, new Mat(), 0, mat3);
            Arrays.sort(scores);
            runTimes--;
        }
        long time_2 = System.currentTimeMillis();
        Vector<Long> result = new Vector<>();
        result.add(time_1);
        result.add(time_2);
        Log.i(TAG,"Retrieval time :"+(time_2-time_1));
        return result;
    }


    public Mat readImage2Mat(){
        Mat m = null;
//        try {
//             m = Utils.loadResource(this.context, R.drawable.pedestrian, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
//        } catch (IOException e){
//            Log.e(TAG,e.toString());
//        }
        return m;
    }


    public void testOpenCVMat() {

        Mat img = new Mat(500, 500, CvType.CV_32F);

        Log.d("OpenCV_Xiao","begin");
        Mat mat1 = new Mat(3, 2, CvType.CV_32F);
        Mat mat2 = new Mat(3, 2, CvType.CV_32F);
        Mat mat3 = new Mat(3, 2, CvType.CV_32F);
        Core.multiply(mat1, mat2, mat3);
        Log.d("OpenCV_Xiao",mat3.dump());
    }

//    static final int SHORTER_SIDE = 256;
//    static final int DESIRED_SIDE = MxNetUtils.IMG_SIZE;
//
//    public static Bitmap processBitmap(final Bitmap origin) {
//        //TODO: error handling
//        final int originWidth = origin.getWidth();
//        final int originHeight = origin.getHeight();
//        int height = SHORTER_SIDE;
//        int width = SHORTER_SIDE;
//        if (originWidth < originHeight) {
//            height = (int)((float)originHeight / originWidth * width);
//        } else {
//            width = (int)((float)originWidth / originHeight * height);
//        }
//        final Bitmap scaled = Bitmap.createScaledBitmap(origin, width, height, false);
//        int y = (height - DESIRED_SIDE) / 2;
//        int x = (width - DESIRED_SIDE) / 2;
//        return Bitmap.createBitmap(scaled, x, y, DESIRED_SIDE, DESIRED_SIDE);
//    }

}

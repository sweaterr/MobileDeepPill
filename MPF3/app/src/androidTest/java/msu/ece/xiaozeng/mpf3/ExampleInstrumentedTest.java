package msu.ece.xiaozeng.mpf3;

import android.content.Context;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import msu.ece.xiaozeng.mpf3.classifier.RefImageDatabase;

import static org.junit.Assert.*;

/**
 * Instrumentation test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    @Test
    public void useAppContext() throws Exception {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getTargetContext();

        assertEquals("msu.ece.xiaozeng.mpf3", appContext.getPackageName());
    }

    @Test
    public void cvMatrix() throws Exception {
        if (OpenCVLoader.initDebug()) {

        }
        Mat A = new Mat(3, 4, CvType.CV_32FC1);
        Mat B = new Mat(4, 4, CvType.CV_32FC1);
        Mat C = new Mat(4, 3, CvType.CV_32FC1);

        A.put(0, 0, new float[]{ 1, 1, 1, 1 });
        A.put(1, 0, new float[]{ 2, 2, 2, 4 });
        A.put(2, 0, new float[]{ 3, 3, 3, 3 });
        System.out.println(A.dump()+"");
    }
}

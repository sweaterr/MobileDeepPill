package msu.ece.xiaozeng.mpf3.classifier;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by xiao on 2/28/17.
 */

public class PillClassifier {

    private static final String TAG = "TFImageClassifier";
    public static final int INPUT_SIZE = 227;
    public static final int OUTPUT_SIZE = 128;

    private static final String MODEL_FILE = "file:///android_asset/frozen_model.pb";

    // Config values.
    private String[] inputNames;



    private float[] floatValues;
    private float[] colorFeature;
    private float[] grayFeature;
    private String[] outputNames;
    private int[] intValues;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param context The context from which to get the asset manager to be used to load assets.
     */
    public PillClassifier(Context context) {
        this(context.getAssets(), MODEL_FILE,  new String[]{"color_ph:0"}, new String[]{"color_fea:0","gray_fea:0"});
    }

    public PillClassifier(AssetManager assetManager, String modelFilename,  String[] inputNames, String[] outputNames) {
        this.inputNames = inputNames;
        this.outputNames = outputNames;

        this.floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];
        this.colorFeature = new float[OUTPUT_SIZE];
        this.grayFeature = new float[OUTPUT_SIZE];
        this.intValues = new int[INPUT_SIZE * INPUT_SIZE];

        this.inferenceInterface = new TensorFlowInferenceInterface();
        this.inferenceInterface.initializeTensorFlow(assetManager, modelFilename);
    }

    public void recognizePill(final Bitmap bitmap) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) ;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF)  ;
            floatValues[i * 3 + 2] = (val & 0xFF) ;
        }

        // Copy the input data into TensorFlow.
        inferenceInterface.fillNodeFloat(
                inputNames[0], new int[]{1, INPUT_SIZE, INPUT_SIZE, 3}, floatValues);

        // Run the inference call.
        inferenceInterface.runInference(outputNames);

        // Copy the output Tensor back into the output array.
        inferenceInterface.readNodeFloat(outputNames[0], colorFeature);
        inferenceInterface.readNodeFloat(outputNames[1], grayFeature);

        Log.d(TAG,"color: "+ Arrays.toString(colorFeature));
        Log.d(TAG,"gray: "+ Arrays.toString(grayFeature));
    }

    public void close() {
        inferenceInterface.close();
    }

}

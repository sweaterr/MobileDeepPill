package msu.ece.xiaozeng.mpf3.classifier;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import msu.ece.xiaozeng.mpf3.entity.Pill;
import msu.ece.xiaozeng.mpf3.util.ArgSort;


/**
 * Created by xiaozeng on 3/1/17.
 */

public class RefImageDatabase {

    private static final String TAG = "RefImageDatabase";
    private final static String FILE_NAME = "ref_db.json";
    private static RefImageDatabase instance;
    private static final int FEA_LENGTH = 128;
    private static final int REF_NUM = 2000;
    private Mat refColorFeature;
    private Mat refGrayFeature;
    private final static double COLOR_WEIGHT = 0.4;
    private final static double GRAY_WEIGHT = 0.6;
    private ArrayList<String> pill_names;

    public static synchronized RefImageDatabase getInstance(Context context) {

        if (instance == null) {
            instance = new RefImageDatabase(context);
        }
        return instance;
    }

    private String fileName;

    private RefImageDatabase(Context context){
        this(context,FILE_NAME);
    }

    private RefImageDatabase(Context context, String fileName){
        this.fileName = fileName;
        String json = null;

        try {
            InputStream is = context.getAssets().open(fileName);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        try {
            JSONObject obj = new JSONObject(json);
            JSONArray m_jArry = obj.getJSONArray("ref_pills");
            pill_names = new ArrayList<>();

            refColorFeature = new Mat(REF_NUM, FEA_LENGTH , CvType.CV_32FC1);
            refGrayFeature = new Mat(REF_NUM, FEA_LENGTH
                    , CvType.CV_32FC1);

            for(int i=0; i<m_jArry.length(); i++) {
                JSONArray pillInfo = m_jArry.getJSONArray(i);

                String pillName = pillInfo.getString(0);
                pill_names.add(pillName);
                JSONArray colorFeaJson = pillInfo.getJSONArray(1);
                JSONArray grayFeaJson = pillInfo.getJSONArray(2);

                double[] colorFea = jsonArray2Array(colorFeaJson);
                double[] grayFea = jsonArray2Array(grayFeaJson);


                refColorFeature.put(i,0,colorFea);
                refGrayFeature.put(i,0,grayFea);

            }

        } catch ( JSONException e){
            e.printStackTrace();
        }
    }


    private  double[] jsonArray2Array(JSONArray jsonArray) throws JSONException {
        double[] listdata = new double[jsonArray.length()];

        if (jsonArray != null) {
            for (int i = 0; i < jsonArray.length(); i++) {
                listdata[i]=jsonArray.getDouble(i) ;
            }
        }

        return listdata;
    }

    private float[] getScore(Mat colorFea, double colorWeight, Mat grayFea, double grayWeight ){
        Mat colorScore = new Mat(2000,1,CvType.CV_32FC1 );
        Core.gemm(refColorFeature, colorFea, colorWeight, new Mat(), 0, colorScore);

        Mat finalScore = new Mat();
        Core.gemm(refGrayFeature, grayFea, grayWeight, colorScore, 1, finalScore);

        float[] finalscores = new float[REF_NUM];
        finalScore.get(0, 0, finalscores);
        return finalscores;
    }

    public ArrayList<Pill> query(Mat colorFea, Mat grayFea){

        float[] score = getScore(colorFea, COLOR_WEIGHT, grayFea, GRAY_WEIGHT);
        int[] rank = ArgSort.getIndicesInOrder(score);
        ArrayList<Pill> candidates = new ArrayList<>();
        for (int i = 0; i<5; i++){
            int r = rank[i];
            String pillName = pill_names.get(r);
            candidates.add(new Pill(pillName));
        }
        return candidates;
    }
}
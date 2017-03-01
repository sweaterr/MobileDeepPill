package msu.ece.xiaozeng.mpf3;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageView;

import com.jph.takephoto.app.TakePhoto;
import com.jph.takephoto.app.TakePhotoImpl;
import com.jph.takephoto.model.CropOptions;
import com.jph.takephoto.model.InvokeParam;
import com.jph.takephoto.model.TContextWrap;
import com.jph.takephoto.model.TResult;
import com.jph.takephoto.permission.InvokeListener;
import com.jph.takephoto.permission.PermissionManager;
import com.jph.takephoto.permission.TakePhotoInvocationHandler;
import com.soundcloud.android.crop.Crop;

import org.opencv.android.OpenCVLoader;

import java.io.File;

import msu.ece.xiaozeng.mpf3.classifier.PillClassifier;
import msu.ece.xiaozeng.mpf3.classifier.TensorFlowImageClassifier;


public class MainActivity extends AppCompatActivity implements TakePhoto.TakeResultListener,InvokeListener {
    private final static String TAG = "MainActivity";
    private PillClassifier pillClassifier;
    private TakePhoto takePhoto;
    private InvokeParam invokeParam;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getTakePhoto().onCreate(savedInstanceState);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        setTitle("MSU Mobile Pill Finder");

        /*initialize tensorflow*/
        if (pillClassifier == null)
            pillClassifier = new PillClassifier(MainActivity.this);

        /*initialize opencv*/
        //if (OpenCVLoader.initDebug()) {
        //}
    }

     public void onClick(View view){

         if (view.getId() == R.id.iv_msu_police_logo){
             Intent intent = new Intent(MainActivity.this , SearchResultActivity.class);
             startActivity(intent);

         } else if (view.getId() == R.id.iv_take_photo){
             File file=new File(Environment.getExternalStorageDirectory(), "/temp/pill.jpg");
             if (!file.getParentFile().exists())file.getParentFile().mkdirs();
             Uri imageUri = Uri.fromFile(file);
             getTakePhoto().onPickFromCaptureWithCrop(imageUri,getCropOptions());
         }
         else if (view.getId() == R.id.iv_start_search){

             File file=new File(Environment.getExternalStorageDirectory(), "/temp/pill.jpg");
             if (!file.exists()){
                 Log.e(TAG,"JPG does not exist!");
                 return;
             }
             else{
                 BitmapFactory.Options bmOptions = new BitmapFactory.Options();
                 Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath(),bmOptions);
                 bitmap = Bitmap.createScaledBitmap(bitmap,PillClassifier.INPUT_SIZE,
                         PillClassifier.INPUT_SIZE, true);
                 pillClassifier.recognizePill(bitmap);
             }


         }

    }

    /**
     * crop option
     * @return
     */
    private CropOptions getCropOptions(){
        int height= 400;
        int width= 400;

        CropOptions.Builder builder=new CropOptions.Builder();
        builder.setAspectX(width).setAspectY(height);
        //builder.setOutputX(width).setOutputY(height);
        builder.setWithOwnCrop(false);
        return builder.create();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onStop() {
        try {
            if (pillClassifier != null) pillClassifier.close();
        } catch (Throwable t) {
            // close quietly
        }
        super.onStop();
    }


    public TakePhoto getTakePhoto(){
        if (takePhoto==null){
            takePhoto= (TakePhoto) TakePhotoInvocationHandler.of(this).bind(new TakePhotoImpl(this,this));
        }
        return takePhoto;
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        getTakePhoto().onSaveInstanceState(outState);
        super.onSaveInstanceState(outState);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        getTakePhoto().onActivityResult(requestCode, resultCode, data);
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionManager.TPermissionType type=PermissionManager.onRequestPermissionsResult(requestCode,permissions,grantResults);
        PermissionManager.handlePermissionsResult(this,type,invokeParam,this);
    }

    @Override
    public void takeSuccess(TResult result) {
        Log.d(TAG,"take success");
    }
    @Override
    public void takeFail(TResult result,String msg) {
        Log.d(TAG,"take Fail");
    }
    @Override
    public void takeCancel() {
        Log.d(TAG,"take Cancel");
    }

    @Override
    public PermissionManager.TPermissionType invoke(InvokeParam invokeParam) {
        PermissionManager.TPermissionType type=PermissionManager.checkPermission(TContextWrap.of(this),invokeParam.getMethod());
        if(PermissionManager.TPermissionType.WAIT.equals(type)){
            this.invokeParam=invokeParam;
        }
        return type;
    }
}

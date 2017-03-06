package msu.ece.xiaozeng.mpf3;

import org.junit.Test;
import static org.junit.Assert.*;
import msu.ece.xiaozeng.mpf3.util.ArgSort;
/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    @Test
    public void addition_isCorrect() throws Exception {
        assertEquals(4, 2 + 2);
    }


    @Test
    public void argsort() throws Exception {
        double[] array = {0.12,0.13,0.12,0.11,0.11,0.11};
        int[] result = ArgSort.getIndicesInOrder(array);
        for(int i =0; i<result.length; i++) {
            System.out.print(result[i]+" ");
        }

    }




}
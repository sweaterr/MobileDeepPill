package msu.ece.xiaozeng.mpf3.util;

/**
 * Created by xiaozeng on 3/6/17.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by chengli on 8/20/14.
 */
public class ArgSort {

    public static int[] getIndicesInOrder(float[] array) {
        Map<Integer, Float> map = new HashMap<Integer, Float>(array.length);
        for (int i = 0; i < array.length; i++)
            map.put(i, array[i]);

        List<Map.Entry<Integer, Float>> l =
                new ArrayList<Map.Entry<Integer, Float>>(map.entrySet());

        Collections.sort(l, new Comparator<Map.Entry<?, Float>>() {
            @Override
            public int compare(Map.Entry<?, Float> e1, Map.Entry<?, Float> e2) {
                return e2.getValue().compareTo(e1.getValue());
            }
        });

        int[] result = new int[array.length];
        for (int i = 0; i < result.length; i++)
            result[i] = l.get(i).getKey();

        return result;
    }


}

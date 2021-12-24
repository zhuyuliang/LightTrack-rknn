# LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search (rknn)

The official implementation by pytorch:

https://github.com/researchmm/LightTrack


# *Run with pre-compiled version*

## Download pre-release(rv1126/rv1109)
https://github.com/Z-Xiong/LightTrack-rknn/releases/tag/rv1109%2Frv1126

### connect your development board
The model only supports rv1109 and rv1126.

### install to rv1126/rv1109
```
$ cd LightTrack-rknn
$ adb push model /userdata
$ adb push 01.mp4 /userdata
$ adb push $PATH_TO_RELEASE/LightTrack /userdata
```

### run by rv1126/rv1109
```
$ adb shell
# in rk shell
$ cd /userdata
$ ./LightTrack 01.mp4
```

### export out.avi
```
# in host bash
$ adb pull /userdata/install/lighttrack_demo/out.avi
```



# *Or you can compile it yourself by following the steps below*


## 0. Download lib
链接: https://pan.baidu.com/s/1iDRuForXsz4p1NNQCZ1QIg  密码: fqba

Unzip the zip
```
$ cp -r lib LightTrack-rknn
```

## 1. How to run it?

### modify your own CMakeList.txt `CROSS_COMPILE_TOOL`
```
set(CROSS_COMPILE_TOOL $YOUR_PATH/rk1109_rk1126_linux/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf)
```

### build
```
$ mkdir build && cd build
$ cmake .. && make -j
$ make install
```

### connect your development board
The model only supports rv1109 and rv1126.

### install to rv1126/rv1109
```
$ adb push ../install /userdata
```

### run by rv1126/rv1109
```
$ adb shell
# in rk shell
$ cd /userdata/install/lighttrack_demo
$ ./LightTrack 01.mp4
```

### export out.avi
```
# in host bash
$ adb pull /userdata/install/lighttrack_demo/out.avi
```

---

# You may encounter build problems, raise an issue.






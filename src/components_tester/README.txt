cloud_clasifier is a python script for examining HSV value distrobutions in a pointcloud.
cloud_clasifier can also be used to extract subsets of a pointcloud based on ranges of hsv values.

Usage:
python cloud_clasifier.py <file> <component to act on (h,s,v,i) <low threashold> <high threashold> <plot the data?(p/n)> <delete points outside range in output graph?(d/n)>

Note: the input graph is not affected and a new file is created with filtered points.


to plot hsv distrobutions:
python cloud_clasifier.py file.ply h 0 1 p n

to delete points outside the desired huegh range:
python cloud_clasifier.py file.ply h 0.2 0.4 n d

to delete points outside the desired value range:
python cloud_clasifier.py file.ply v 0.2 0.4 n d

to delete points outside the desired huegh & value range:
python cloud_clasifier.py file.ply hv 0.2 0.4 n d


to plot the origional distrobution and delete points outside the desired value range:
python cloud_clasifier.py file.ply v 0.2 0.4 p d


import sys
import argparse
import colorsys

from plyfile import PlyData, PlyElement

import numpy as np
from matplotlib import pyplot as plt

# generate numpy histogram of input data
def plot_bins(data, name):
    bins = np.arange(0.0,1.0,0.005)
    plt.hist(data, bins=bins)
    plt.title(name)
    plt.show()

#Load a pointcloud from .ply file display histograms of hsv values and delete points values outside the specified range
def main(args):
    #load a pointcloud from .ply file (points only, ignores mesh connections (untested on mesh as .ply file))
    plydata = PlyData.read(args.file+'.ply')
    high = args.high
    low = args.low
    h = []
    l = []
    s = []
    v = []

    removal = []
    count = 0

    #read each point
    for vert in plydata['vertex']:
        #get the hsv and hls colorspace values (note this converts from 8 bit from file to floating point for python
        hsv = colorsys.rgb_to_hsv(vert['red']/255.00, vert['green']/255.00, vert['blue']/255.00) 
        hls = colorsys.rgb_to_hls(vert['red']/255.00, vert['green']/255.00, vert['blue']/255.00) 

        #make lists of seperate values for graphing later
        h.append(hsv[0])
        l.append(hls[1])
        s.append(hsv[1])
        v.append(hsv[2])
        

        #focus on components we care about only, prep a list of points to remove
        if('h' in args.component and (hsv[0] < low or hsv[0] > high)):
            removal.append(count)
        if('s' in args.component and (hsv[1] < low or hsv[1] > high)):
            removal.append(count)
        if('v' in args.component and (hsv[2] < low or hsv[2] > high)):
            removal.append(count)
        if('l' in args.component and (hls[1] < low or hls[1] > high)):
            removal.append(count)
        count += 1


    #plot points if plotting was specified
    if ('p' in args.plot):
        plot_bins(h, 'huegh')
        plot_bins(l, 'lightness')
        plot_bins(s, 'saturation')
        plot_bins(v, 'value')
    #delete points outside range if deleting was specified
    if ('d' in args.delete):
        plydata['vertex'].data = np.delete(plydata['vertex'].data, removal)
    plydata.write(args.file+'_out.ply')

    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='extract features from pointcloud by huegh.')
    parser.add_argument('file', action='store', type=str, help='file name.') 
    parser.add_argument('component', action='store', type=str, help='components to use for isolation') 
    parser.add_argument('low', action='store', type=float, help='min huegh value') 
    parser.add_argument('high', action='store', type=float, help='max huegh value') 
    parser.add_argument('plot', action='store', type=str, help='plot the figures?') 
    parser.add_argument('delete', action='store', type=str, help='delete selected data') 

    args = parser.parse_args()
    #check for .ply file and strip extension from name pass to main
    if '.ply' in args.file:
        args.file = args.file[:args.file.find('.ply')]
    main(args)


# Copyright (c) 2013 The University of Edinburgh.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 


#!/usr/bin/python
# Plotter for OpenACC Benchmarks benchmarks

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

class Globs:
    """Class for storing objects such as the session dictionary and a counter."""
    def __init__(self):
        self.colour = False
        self.inputfile = 'data.txt'
        self.debug = False
        self.eps = False

def chomp(s):
    return s[:-1] if s.endswith('\n') else s


def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': None, 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }

    for line in ax.get_lines():
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)


def main():

    # Make a singular global instance for holding odd global values
    global GL
    GL = Globs()

    # Parse the input arguements in a nice manner
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Enable debug outputs.", action="store_true")
    parser.add_argument("-i", "--input", help="Input file to read (default: ./data.txt).")
    parser.add_argument("-c", "--colour", help="Colour graphs. (default: Black + White).", action="store_true")
    parser.add_argument("-e", "--eps", help="Output graphs in Enhanced Postscript Format (.eps) rather than JPEG.", action="store_true")
    args = parser.parse_args()

    if args.debug:
        GL.debug = True

    if args.colour:
        GL.colour = True

    if args.input:
        GL.inputfile = args.input

    if args.eps:
        GL.eps = True


    # Open input file, consume file, close.
    inp = open(GL.inputfile, 'r')
        
    data_array = []
    for l in inp:
        data_array.append(chomp(l).split())
    inp.close()

    names = []
    compilers = []
    datasizes = []
            
    for line in data_array:
        names.append(line[1])
        compilers.append(line[0])
        datasizes.append(int(line[2]))
                

    # Unique-ify these lists   
    names = list(set(names))
    compilers = list(set(compilers))
    datasizes = list(set(datasizes))
                
    ds = np.zeros( (len(datasizes),1) )
    datasizes = sorted(datasizes)
    for i in range(0,len(datasizes)):
        ds[i][0] = int(datasizes[i])
                   


                    
# Loop over the tests
# Create and array of times for each one and add data as necessary
# Then plot for each test

    if GL.debug:
        print names
        print datasizes
        print compilers

    for i in names:
        times = np.zeros( (len(datasizes),len(compilers)) )
        for c in compilers:
            for r in range(0, len(data_array)):
                if (data_array[r][0] == c and data_array[r][1] == i):
                    times[datasizes.index(int(data_array[r][2]))][compilers.index(c)] = abs(float(data_array[r][3]))
                
        ds2 = np.log(ds)/np.log(2)
        unit = r'$\mu s$'

        if np.amax(times) > 1000:
            times = times / 1000
            unit = r'$ms$'


        if np.amax(times) > 1000000:
            times = times / 1000000
            unit = r'$s$'  

        fig = plt.figure()
        plt.semilogy(ds2,times,'-', linewidth=3)
        plt.xlabel('Datasize (M Bytes)', size='large')
        if i=='Kernels_combined' or i=='Kernels_If' or i=='Parallel_If' or i=='Parallel_private' or i=='Parallel_firstprivate' or i=='Parallel_reduction' or i=='Kernels_reduction' or i=='Update_Host' or i=='Kernels_Invocation':
            plt.ylabel('Difference (' + unit + ')', size='large')
        else:
            plt.ylabel('Run time (' + unit + ')', size='large')

        locs, labs = plt.xticks()
        lmax = max(locs)+1
        lmin = min(locs)
        
        plt.xticks(np.arange(lmin,lmax), ('1','2','4','8','16','32','64','128','256','512','1024'),size='large') 
        xmin,xmax = plt.xlim()
        plt.xlim(xmin*0.99,xmax*1.01)
        plt.yticks(size='large')
        i = i.replace('_',' ')
        plt.title(i, size='large')
        if GL.colour == False:
            setFigLinesBW(fig)

        comp_normalized = []
        for c in compilers:
            c = c.replace('_', ' ')
            comp_normalized.append(c)
        plt.legend(comp_normalized, loc='best')
        if GL.eps == True:
            plt.savefig(i+'.eps', dpi=660, bbox_inches='tight')
        else:
            plt.savefig(i+'.jpg', bbox_inches='tight')
        plt.close()                                  


if __name__ == "__main__":
    sys.exit(main())

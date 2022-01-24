import pandas as pd
import numpy as np
import yfinance
import fractions
import matplotlib.pyplot as plt
import matplotlib as mpl

class LT3:
    # __ class for a particualr 3x3 linear transform (with specified input vector) 
    # __ calculates: A) the transformed vector (self.b_trans),
    #                B) the normal vector to the [vector, transformed_vector] plane (self.normal)
    #                C) the angle between the original vector and the transformed one (self.angle)

    np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())}) # forces display in rationals
    def __init__(self, M, b):
        self.M = M
        self.b = b
        self.b_trans = self.solve()
        self.normal, self.angle = self.extra_solves()


    def solve(self):
        # __ calculates inverse of input matrix, uses this to compute solution to Mx=b
        M_inv = np.linalg.inv(self.M)
        return np.dot(M_inv,self.b)


    def extra_solves(self):
        # __ computes plane normal vector and angle betweeen input vector and transform
        n_b = self.norm(self.b)
        n_b_trans = self.norm(self.b_trans)
        plane_normal = self.plane_normal(self.b, self.b_trans)
        angle = np.arccos(np.dot(self.b, self.b_trans))
        return plane_normal, angle

    def norm(self, v):
        # __ returns norm of vector
        return np.sqrt(np.sum(np.dot(v,v)))

    def plane_normal(self, a, b):
        # __ returns (unit) normal vector to plane spanned by a and b (chooses right hand convention by use of cross product)
        v = np.cross(a,b)
        return v/self.norm(v)
        

        
class PLOTT:
    def __init__(self,m,b):
        # __ calculates orthogonal vectors to input; first orthongonal vector calc is normal to plane of original and transformed vector,
        # __ following calc from subsequent plane. This provides an easy choice of basis.
        self.M = m

        self.b1 = b
        self.LT1 = LT3(self.M, self.b1)
        self.b1t = self.LT1.b_trans

        self.b2 = self.LT1.normal*self.norm(self.b1)
        self.LT2 = LT3(self.M, self.b2)
        self.b2t = self.LT2.b_trans

        self.b3 = self.plane_normal(self.b1, self.b2)*self.norm(self.b1)
        self.LT3 = LT3(self.M, self.b3)
        self.b3t = self.LT3.b_trans

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.format_plot()

        self.initial_volume = self.volume(self.b1,self.b2,self.b3)
        self.trans_volume = self.volume(self.b1t,self.b2t,self.b3t)


    def format_plot(self):   
        # __ formats for plotting
        self.ax.view_init(elev=30, azim=2)
        ax_lim = self.find_ax_lim()
        self.ax.set_xlim3d(-ax_lim, ax_lim)
        self.ax.set_ylim3d(-ax_lim, ax_lim)
        self.ax.set_zlim3d(-ax_lim, ax_lim)



    def plott(self):
        # __ plots rhomboid of orthogonal inputs and corresponding transformed outputs

        endp = self.b1+self.b2+self.b3
        endp_trans = self.b1t+self.b2t+self.b3t
        print(endp.shape)

        plt.quiver(0,0,0,*list(self.b1), arrow_length_ratio=0.01, color='b')
        plt.quiver(0,0,0,*list(self.b1t), color='b', arrow_length_ratio=0.01)
        
        plt.quiver(*list(self.b1), *list(endp-self.b1), arrow_length_ratio=0.01)
        plt.quiver(*list(self.b1t), *list(endp_trans-self.b1t), color='r', arrow_length_ratio=0.01)

        plt.quiver(0,0,0,*list(self.b2), arrow_length_ratio=0.01)
        plt.quiver(0,0,0,*list(self.b2t), color='r', arrow_length_ratio=0.01)
        
        plt.quiver(*list(self.b2), *list(endp-self.b2), arrow_length_ratio=0.01)
        plt.quiver(*list(self.b2t), *list(endp_trans-self.b2t), color='r', arrow_length_ratio=0.01)

        plt.quiver(0,0,0,*list(self.b3), arrow_length_ratio=0.01)
        plt.quiver(0,0,0,*list(self.b3t), color='r', arrow_length_ratio=0.01)
        
        plt.quiver(*list(self.b3), *list(endp-self.b3), arrow_length_ratio=0.01)
        plt.quiver(*list(self.b3t), *list(endp_trans-self.b3t), color='r', arrow_length_ratio=0.01)

        plt.quiver(*list(self.b1), *list(self.b2-self.b1), arrow_length_ratio=0.01, color='b')
        plt.quiver(*list(self.b1), *list(self.b3-self.b1), arrow_length_ratio=0.01)
        plt.quiver(*list(self.b2), *list(self.b3-self.b2), arrow_length_ratio=0.01)

        plt.quiver(*list(self.b1t), *list(self.b2t-self.b1t), arrow_length_ratio=0.01, color='b')
        plt.quiver(*list(self.b1t), *list(self.b3t-self.b1t), arrow_length_ratio=0.01, color='r')
        plt.quiver(*list(self.b2t), *list(self.b3t-self.b2t), arrow_length_ratio=0.01, color='r')
        

    def find_ax_lim(self):
        # __ finds axes limits for plotting
        return 2*max([np.max(self.b1), np.max(self.b1t)])

    def norm(self, v):
        # __ returns norm of vector
        return np.sqrt(np.sum(np.dot(v,v)))

    def plane_normal(self, a, b):
        # __ returns (unit) normal vector to plane spanned by a and b (chooses right hand convention by use of cross product)
        v = np.cross(a,b)
        return v/self.norm(v)

    def volume(self, a, b, c):
        # __ calculates volume of parallelogram spanned by a, b, c 
        return np.dot(np.cross(a,b),c)
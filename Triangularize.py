

import os
import cv2
import random
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from tkinter.filedialog import askopenfilename
from scipy.spatial import Delaunay, Voronoi


def load_image():
    """
    Load an image file using tkinter GUI.
    """

    # Create tkinter window and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Create window to ask for file path
    path = askopenfilename()
    root.destroy()

    # Return loaded image array
    return cv2.imread(path, 1)

    
def random_point_cloud(image, n_pts):
    """
    Generate a series of random points to lie within the frame of an image.
    """

    height, width = image.shape

    rand_pts = []
    for i in range(n_pts):
        rand_pt = [random.randint(1, width-1), random.randint(1, height-1)]
        rand_pts.append(rand_pt)

    return np.array(rand_pts)


def generate_frame_points(height, width, n_pts):
    """
    Generate points on window frame. @param n_pts is the number of points on one edge of the image.
    """

    l_edge_y = np.arange(0, height, height // n_pts)
    l_edge_x = np.zeros(l_edge_y.size)
    l_edge = np.column_stack((l_edge_x, l_edge_y))


    r_edge_y = np.arange(0, height, height // n_pts)
    r_edge_x = np.zeros(r_edge_y.size) + width
    r_edge = np.column_stack((r_edge_x, r_edge_y))


    t_edge_x = np.arange(width/n_pts , width - width//n_pts, width//n_pts)  # Avoid overlapping points on corners
    t_edge_y = np.zeros(t_edge_x.size)
    t_edge = np.column_stack((t_edge_x, t_edge_y))


    b_edge_x = np.arange(width/n_pts , width - width//n_pts, width//n_pts)  
    b_edge_y = np.zeros(t_edge_x.size) + height
    b_edge = np.column_stack((b_edge_x, b_edge_y))
    

    return np.concatenate((l_edge, r_edge, t_edge, b_edge))


def draw_feature_points(image, feature_points):
    """
    Draw feature points from a (array.size//2, 2) format array
    """
    image_features = image.copy()
    for point in feature_points:
        image_features = cv2.circle(image_features,
                                    center=(point[0], point[1]),
                                    radius=5, color=(0,0,255),
                                    thickness=-1)
    return image_features


def generate_feature_points(image, method='good', **kwargs):

    height, width, channels = image.shape
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'good':
        feature_points = cv2.goodFeaturesToTrack(image_gray,
                                                 maxCorners=1000,
                                                 qualityLevel=0.01,
                                                 minDistance=10)[::, 0, ::]  # Reshape to proper (array_size//2, 2) shape

    if method == 'random':
        feature_points = random_point_cloud(image_gray, n_pts=200)

    frame_points = generate_frame_points(height, width, 20)
    feature_points = np.concatenate((feature_points, frame_points))

    return feature_points


def main():
##    #image = load_image()
##    #image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
##
##    feature_points = generate_feature_points(image, method='good')
##
##    del_tesselation = Delaunay(feature_points)
##    triangles = feature_points[del_tesselation.simplices]
##
##    #voronoi_points = Voronoi(feature_points).vertices
##
##    image_triangles = image.copy()
##    for vertices in triangles:
##
##        centroid_x = int(np.mean(vertices[:,0]))
##        centroid_y = int(np.mean(vertices[:,1]))
##
##        centroid_color = tuple(image_triangles[centroid_y, centroid_x].tolist())  # Convert array to tuple
##        image_triangles = cv2.fillPoly(image_triangles, np.int32([vertices]), color=centroid_color)
##
##    cv2.imshow("", image_triangles)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
    #cv2.imwrite("opencv_random.png", image_triangles)
    
##    plt.figure(frameon=False)
##    plt.triplot(feature_points[:,0], feature_points[:,1], del_tesselation.simplices)
##    plt.plot(feature_points[:,0], feature_points[:,1], 'yo', markersize=2)
##    plt.plot(voronoi_points[:,0], voronoi_points[:,1], 'ro', markersize=2)
##    plt.plot(centroids[:,0], centroids[:,1], 'ro', markersize=2)
##    plt.imshow(image_RGB)
##    plt.axis("off")
##    plt.savefig("matplotlib.png", bbox_inches='tight', pad_inches=0)
##    plt.show()

    
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, image = cap.read()
        if ret==True:
            feature_points = generate_feature_points(image, method='good')

            del_tesselation = Delaunay(feature_points)
            triangles = feature_points[del_tesselation.simplices]
            
            image_triangles = image.copy()
            for vertices in triangles:

                centroid_x = int(np.mean(vertices[:,0]))
                centroid_y = int(np.mean(vertices[:,1]))

                centroid_color = tuple(image_triangles[centroid_y, centroid_x].tolist())  # Convert array to tuple
                image_triangles = cv2.fillPoly(image_triangles, np.int32([vertices]), color=centroid_color)

            image_triangles = cv2.resize(image_triangles, None, fx=2., fy=2.)
            cv2.imshow('frame',image_triangles)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

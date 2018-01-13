import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

def apply_gauss(img, width):
    # check if width is odd number
    if (width%2 == 0):
        raise ValueError('width parameter must be an odd number')
    # given filter
    g1 = np.array([[1,2,1]])
    g1 = g1 / np.sum(g1)
    filt = np.copy(img)
    
    # row-filtering, followed by column filtering     
    for i in range(width):
        filt = np.apply_along_axis(np.convolve, 1, filt, np.ravel(g1), mode="same")
        filt = np.apply_along_axis(np.convolve, 0, filt, np.ravel(g1), mode="same")

    return filt

def image_blurring(filename, width):
	img = scipy.ndimage.imread(filename, flatten=False, mode=None)
	plt.imshow(img, cmap='gray')
	plt.title('original')
	plt.show()
	filtered = apply_gauss(img, width)
	plt.imshow(filtered, cmap='gray')
	plt.title('filtered')
	plt.show()

def getDescriptors(filename1, filename2):
	img1 = cv2.imread(filename1)
	img2 = cv2.imread(filename2)

	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(gray1,None)
	kp2, des2 = sift.detectAndCompute(gray2,None)

	img1=cv2.drawKeypoints(gray1,kp1,None)
	img2=cv2.drawKeypoints(gray2,kp2,None)

	# cv2.imwrite('scene_sift_keypoints.jpg',img1)
	# cv2.imwrite('book_sift_keypoints.jpg',img2)

	print("number of regions in book.pgm: ", len(des1))
	print("number of regions in scene.pgm: ", len(des2))
	print("shape of each descriptor vector: ", des1[0].shape)
	plt.imshow(img1),plt.show()
	plt.imshow(img2),plt.show()
	return (kp1, des1, kp2, des2, img1, img2, gray1, gray2)

def getMatches(des1, des2, kp1, kp2, img1, img2):
	# img1 = cv2.imread(filename1)
	# img2 = cv2.imread(filename2)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	coords = []

	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.9*n.distance:
	        good.append([m])
	        idx1 = m.queryIdx
	        idx2 = m.trainIdx
	        pt1 = kp1[idx1].pt
	        pt2 = kp2[idx2].pt
	        if ( ((pt1, pt2)) not in coords):
	            coords.append((pt1, pt2))

	# cv2.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2,outImg=None)
	# cv2.imwrite('matches.png',img3)
	plt.imshow(img3),plt.show()
	return coords

def get3RandPoints(coords):
    randNums = []
    for i in range(3):
        this_rand = np.random.randint(0,len(coords))
        while this_rand in randNums: #this ensures the same set of coordinates are not picked twice
            this_rand = np.random.randint(0,len(coords))
        randNums.append(this_rand)
    randCoords = []
    
    for idx in randNums:
        randCoords.append(coords[idx])
    
    return randCoords

def constructMatrices(randCoords):
    x = np.zeros((6,6))
    x_prime = np.zeros((6,1))
        
    for i in range(6):        
        x_c = randCoords[int(i/2)][0][0]
        y_c = randCoords[int(i/2)][0][1]
        xp_c = randCoords[int(i/2)][1][0]
        yp_c = randCoords[int(i/2)][1][1]
        if(i%2 == 0):
            x[i][0] = x_c
            x[i][1] = y_c
            x[i][4] = 1
        else:
            x[i][2] = x_c
            x[i][3] = y_c
            x[i][5] = 1

        if(i%2 == 0):
            x_prime[i] = xp_c
            x_prime[i+1] = yp_c
    
    return (x, x_prime)

def getAffTrans(x, x_prime):
    transformation = np.linalg.solve(x, x_prime)
    m = np.asarray([[transformation[0][0],transformation[1][0]],[transformation[2][0],transformation[3][0]]])
    t = np.asarray([transformation[4],transformation[5]])
    return (m, t)

def getBestM(coords):
	max_radius = 10.0
	max_inliers = 0
	best_M = np.zeros((2,3))

	for n in range(100):
	    randCoords = get3RandPoints(coords)
	    x, x_prime = constructMatrices(randCoords)
	    m, t = getAffTrans(x, x_prime)
	    this_inliers = 0
	    
	    for item in coords:
	        x_p = item[1][0]
	        y_p = item[1][1]
	        x_t = item[0][0]
	        y_t = item[0][1]
	        actual_pos = np.asarray((x_p, y_p))
	        new_pos = (np.dot(m, np.asarray([ [x_t], [y_t]])) + t).T
	        if (np.absolute(np.linalg.norm(new_pos - actual_pos)) < max_radius):
	            this_inliers += 1
	            
	    if (this_inliers > max_inliers):
	        best_M = np.hstack((m, t))
	        max_inliers = this_inliers
	    
	print("max_inliers", max_inliers)
	print("best_M: \n", best_M)
	return best_M

def affineTrans(best_M, gray1, gray2):
	rows,cols = gray1.shape
	dst = cv2.warpAffine(gray1,best_M,(cols,rows))
	plt.imshow(gray2, cmap='gray')
	plt.title("Actual")
	plt.show()
	plt.subplot(121),plt.imshow(gray1, cmap='gray'),plt.title('Input')
	plt.subplot(122),plt.imshow(dst, cmap='gray'),plt.title('Output')
	plt.show()


def image_alignment(filename1, filename2):
	kp1, des1, kp2, des2, img1, img2, gray1, gray2 = getDescriptors(filename1, filename2)
	coords = getMatches(des1, des2, kp1, kp2, img1, img2)
	best_M = getBestM(coords)
	affineTrans(best_M, gray1, gray2)

def homogeneousCoords(image, world):
	image_h = np.vstack((image, np.ones((1,image.shape[1]))))
	world_h = np.vstack((world, np.ones((1,world.shape[1]))))
	return (image_h, world_h)

def getA(image_h, world_h):
	A = np.zeros((20,12))

	for i in range(image_h.shape[1]):
	    x = image_h.T[i][0]
	    y = image_h.T[i][1]
	    w = image_h.T[i][2]
	    
	    x_world = x * world_h.T[i]
	    y_world = y * world_h.T[i]
	    w_world = w * world_h.T[i]
	    
	    A[i*2][4:8] = -w_world
	    A[i*2][8:12] = y_world
	    A[i*2 + 1][0:4] = w_world
	    A[i*2 + 1][8:12] = -x_world

	return A

def getAndVerifyP(A, world_h, image):
	p = np.linalg.svd(A)[2][-1]
	P = p.reshape((3,4))
	print("P: \n", P)
	zero_prod = np.dot(A, p)
	avg_zero_prod_error = np.average( zero_prod - np.zeros((zero_prod.shape)))
	print("average error in A.p calculation:", avg_zero_prod_error)
	img_calc_hom = np.dot(P, world_h) 
	img_calc_cart = np.asarray([ img_calc_hom[0]/img_calc_hom[2], img_calc_hom[1]/img_calc_hom[2] ])
	avg_projection_error = np.average(np.abs(img_calc_cart - image))
	print("average world-to-image projection error:", avg_projection_error)
	return P

def getAndVerifyC(P):
	C = np.linalg.svd(P)[2][-1]
	zero_vec = np.dot(P, C)
	print("average error in PC=0: ", np.average(np.absolute(zero_vec)))
	print("C_homogenous: \n", C)
	C_inhom = np.asarray(([ C[0]/C[3], C[1]/C[3], C[2]/C[3] ]))
	return C_inhom


def camParams(filename1, filename2):
	image = np.loadtxt(filename1)
	world = np.loadtxt(filename2)
	image_h, world_h = homogeneousCoords(image, world)
	A = getA(image_h, world_h)
	P = getAndVerifyP(A, world_h, image)
	C = getAndVerifyC(P)
	print("C: \n", C)

def getCenters(image_points):
	x_centers = np.mean(image_points[0], axis=0)
	y_centers = np.mean(image_points[1], axis=0)
	centers = np.vstack((x_centers, y_centers))
	return centers

def getW(image_points, centers):
	c_image_points = np.copy(image_points)
	for i in range(len(image_points[0])):
		c_image_points[0][i] -= centers[0]
		c_image_points[1][i] -= centers[1]

	W = np.vstack((c_image_points[0].T, c_image_points[1].T))
	return W

def showStructMotResults(W, centers):
	U, D, V = np.linalg.svd(W)
	M_i_matrix = np.dot(U[:,0:3], np.diag(D[0:3]))
	print("M1: \n", M_i_matrix[0:2,:])
	print("t1: \n", centers[:,0])
	print("3d coords of first 10 world points: \n",  V.T[0:10,0:3])
	x_s = V.T[:,0]
	y_s = V.T[:,1]
	z_s = V.T[:,2]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xs=x_s, ys=y_s, zs=z_s)
	plt.show()

def structFromMot(filename1):
	sfm_points = scipy.io.loadmat(filename1)
	image_points = sfm_points['image_points']
	centers = getCenters(image_points)
	W = getW(image_points, centers)
	showStructMotResults(W, centers)



print("problem 1: image filtering")
image_blurring("./assignment1/parrot_grey.png", 3)
print("problem 2: image alignment")
image_alignment("./assignment1/book.pgm", "./assignment1/scene.pgm")
print("problem 3: estimating camera parameters")
camParams("./assignment1/image.txt", "./assignment1/world.txt")
print("problem 4: structure from motion")
structFromMot("./assignment1/sfm_points.mat")

